#!/usr/bin/env python3
"""
Real-time Log Anomaly Detection System
=====================================

A modular, production-ready system for detecting anomalies in JSONL log files
using machine learning algorithms with real-time monitoring capabilities.

Features:
- Real-time log file monitoring with watchdog
- Multiple ML algorithms: Isolation Forest, LOF, One-Class SVM
- Configurable alerting (console, webhook, syslog, file)
- Batch processing with configurable windows
- Model persistence and loading
- Performance optimized with async processing

Usage:
    python anomaly_detector.py --input logs/live_logs.jsonl --algorithm all
    python anomaly_detector.py --input logs/live_logs.jsonl --algorithm isolation_forest --contamination 0.05
    python anomaly_detector.py --input logs/live_logs.jsonl --webhook-url http://alerts.example.com/webhook
"""

import argparse
import asyncio
import json
import logging
import pickle
import sys
import time
import syslog
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from queue import Queue
import threading

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Try to import colorama, but don't fail if it's not available
try:
    from colorama import init as colorama_init
except ImportError:
    def colorama_init(autoreset=True):
        pass


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LogEntry:
    """Represents a single log entry with metadata."""
    
    def __init__(self, raw_line: str, line_number: int, timestamp: Optional[str] = None):
        self.raw_line = raw_line.strip()
        self.line_number = line_number
        self.timestamp = timestamp or datetime.now().isoformat()
        self.log_id = f"log_{line_number}_{int(time.time())}"
        self.parsed_data = self._parse_json()
    
    def _parse_json(self) -> Dict[str, Any]:
        """Parse JSONL entry, return empty dict if invalid."""
        try:
            return json.loads(self.raw_line)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON at line {self.line_number}")
            return {}


class FeatureExtractor:
    """Extracts numerical features from log entries for ML algorithms."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.text_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.is_fitted = False
        self.feature_names = []
        self.expected_numeric_features = 5  # Number of base numeric features
    
    def extract_features(self, log_entries: List[LogEntry]) -> np.ndarray:
        """Extract features from log entries."""
        if not log_entries:
            return np.array([]).reshape(0, -1)
        
        features_list = []
        text_data = []
        
        for entry in log_entries:
            numeric_features = self._extract_numeric_features(entry.parsed_data)
            features_list.append(numeric_features)
            text_data.append(self._extract_text_features(entry.parsed_data))
        
        # Ensure consistent feature dimensions
        max_numeric_features = max(len(f) for f in features_list) if features_list else 0
        max_numeric_features = max(max_numeric_features, self.expected_numeric_features)
        
        # Pad numeric features to consistent length
        for i, features in enumerate(features_list):
            if len(features) < max_numeric_features:
                features_list[i] = features + [0.0] * (max_numeric_features - len(features))
            elif len(features) > max_numeric_features:
                features_list[i] = features[:max_numeric_features]
        
        numeric_features = np.array(features_list)
        
        if not self.is_fitted:
            # Fit vectorizer and scaler on first batch
            if text_data and any(text_data):
                try:
                    text_features = self.text_vectorizer.fit_transform(text_data).toarray()
                except ValueError:
                    # Handle case where all text is empty
                    text_features = np.zeros((len(log_entries), 100))
            else:
                text_features = np.zeros((len(log_entries), 100))
            
            if numeric_features.size > 0 and numeric_features.shape[0] > 0:
                self.scaler.fit(numeric_features)
                numeric_features = self.scaler.transform(numeric_features)
            
            self.is_fitted = True
        else:
            # Transform using fitted vectorizer and scaler
            if text_data and any(text_data):
                try:
                    text_features = self.text_vectorizer.transform(text_data).toarray()
                except ValueError:
                    # Handle case where vocabulary is empty
                    text_features = np.zeros((len(log_entries), 100))
            else:
                text_features = np.zeros((len(log_entries), 100))
            
            if numeric_features.size > 0 and numeric_features.shape[0] > 0:
                numeric_features = self.scaler.transform(numeric_features)
        
        # Combine features
        if numeric_features.size > 0 and text_features.size > 0:
            combined_features = np.hstack([numeric_features, text_features])
        elif numeric_features.size > 0:
            combined_features = numeric_features
        else:
            combined_features = text_features
        
        return combined_features
    
    def _extract_numeric_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract numeric features from parsed JSON data."""
        features = []
        
        # Common log features
        features.append(len(str(data.get('message', ''))))  # Message length
        features.append(len(data))  # Number of fields
        features.append(float(data.get('response_time', 0)) if isinstance(data.get('response_time'), (int, float)) else 0.0)
        features.append(float(data.get('status_code', 200)) if isinstance(data.get('status_code'), (int, float)) else 200.0)
        features.append(float(data.get('content_length', 0)) if isinstance(data.get('content_length'), (int, float)) else 0.0)
        
        # Extract all numeric values (limited to prevent dimension explosion)
        numeric_count = 0
        for value in data.values():
            if isinstance(value, (int, float)) and numeric_count < 15:
                features.append(float(value))
                numeric_count += 1
        
        return features
    
    def _extract_text_features(self, data: Dict[str, Any]) -> str:
        """Extract text content for TF-IDF vectorization."""
        text_parts = []
        
        for key, value in data.items():
            if isinstance(value, str):
                text_parts.append(value)
        
        return ' '.join(text_parts) if text_parts else ''


class AnomalyDetector:
    """Wrapper for different anomaly detection algorithms."""
    
    def __init__(self, algorithm: str = 'isolation_forest', contamination: float = 0.1):
        self.algorithm = algorithm
        self.contamination = contamination
        self.models = {}
        self.is_fitted = False
        self.training_data = []
        self.min_samples_for_training = 50  # Minimum samples before training
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models."""
        if self.algorithm in ['isolation_forest', 'all']:
            self.models['isolation_forest'] = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
        
        if self.algorithm in ['lof', 'all']:
            # Adaptive n_neighbors based on expected data size
            n_neighbors = min(20, max(5, int(self.min_samples_for_training * 0.1)))
            self.models['lof'] = LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True,
                n_neighbors=n_neighbors
            )
        
        if self.algorithm in ['one_class_svm', 'all']:
            self.models['one_class_svm'] = OneClassSVM(
                gamma='scale',
                nu=self.contamination
            )
    
    def accumulate_training_data(self, X: np.ndarray):
        """Accumulate training data until we have enough samples."""
        if X.shape[0] > 0:
            self.training_data.append(X)
            total_samples = sum(arr.shape[0] for arr in self.training_data)
            
            if total_samples >= self.min_samples_for_training and not self.is_fitted:
                combined_data = np.vstack(self.training_data)
                self._fit_models(combined_data)
                self.is_fitted = True
                # Clear training data to save memory
                self.training_data = []
                return True
        return False
    
    def _fit_models(self, X: np.ndarray):
        """Fit the anomaly detection models."""
        if X.shape[0] == 0:
            logger.warning("No data to fit models")
            return
        
        logger.info(f"Training models with {X.shape[0]} samples")
        failed_models = []
        
        for name, model in self.models.items():
            try:
                if name == 'lof' and X.shape[0] < model.n_neighbors:
                    logger.warning(f"Skipping LOF training: need at least {model.n_neighbors} samples, got {X.shape[0]}")
                    failed_models.append(name)
                    continue
                    
                logger.info(f"Fitting {name} model")
                model.fit(X)
                logger.info(f"✅ {name} model trained successfully")
            except Exception as e:
                logger.error(f"❌ Failed to fit {name}: {e}")
                failed_models.append(name)
        
        # Remove failed models
        for name in failed_models:
            if name in self.models:
                del self.models[name]
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict anomalies using fitted models."""
        results = {}
        
        if X.shape[0] == 0 or not self.is_fitted:
            return results
        
        for name, model in list(self.models.items()):  # Use list() to avoid dict change during iteration
            try:
                if name == 'lof':
                    # LOF requires at least 1 sample for prediction in novelty mode
                    if X.shape[0] < 1:
                        logger.warning(f"Skipping LOF prediction: need at least 1 sample, got {X.shape[0]}")
                        continue
                    predictions = model.predict(X)
                    scores = model.decision_function(X)
                else:
                    predictions = model.predict(X)
                    if hasattr(model, 'score_samples'):
                        scores = model.score_samples(X)
                    else:
                        scores = predictions.astype(float)
                
                results[name] = {
                    'predictions': predictions,
                    'scores': scores
                }
            except Exception as e:
                logger.error(f"❌ Prediction failed for {name}: {e}")
        
        return results
    
    def save_models(self, path: str):
        """Save trained models to disk."""
        if not self.is_fitted:
            logger.warning("No trained models to save")
            return
            
        model_data = {
            'models': self.models,
            'algorithm': self.algorithm,
            'contamination': self.contamination,
            'is_fitted': self.is_fitted
        }
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Models saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def load_models(self, path: str):
        """Load pre-trained models from disk."""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.algorithm = model_data['algorithm']
            self.contamination = model_data['contamination']
            self.is_fitted = model_data.get('is_fitted', False)
            logger.info(f"Models loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")


class AlertManager:
    """Handles different types of alerting mechanisms."""
    
    def __init__(self, webhook_url: str = None, alert_file: str = None, use_syslog: bool = False):
        self.webhook_url = webhook_url
        self.alert_file = alert_file
        self.use_syslog = use_syslog
        
        if self.use_syslog:
            try:
                syslog.openlog("anomaly_detector")
            except Exception as e:
                logger.error(f"Failed to initialize syslog: {e}")
                self.use_syslog = False
    
    def send_alert(self, anomaly_data: Dict[str, Any]):
        """Send alert through configured channels."""
        alert_message = self._format_alert(anomaly_data)
        
        # Console output (always enabled)
        print(f"🚨 ANOMALY DETECTED: {alert_message}")
        
        # Webhook alert
        if self.webhook_url:
            self._send_webhook(anomaly_data)
        
        # Syslog alert
        if self.use_syslog:
            try:
                syslog.syslog(syslog.LOG_WARNING, alert_message)
            except Exception as e:
                logger.error(f"Failed to send syslog alert: {e}")
        
        # File alert
        if self.alert_file:
            self._write_to_file(anomaly_data)
    
    def _format_alert(self, anomaly_data: Dict[str, Any]) -> str:
        """Format alert message."""
        return (f"Log ID: {anomaly_data['log_id']}, "
                f"Score: {anomaly_data['anomaly_score']:.3f}, "
                f"Algorithm: {anomaly_data['algorithm']}")
    
    def _send_webhook(self, anomaly_data: Dict[str, Any]):
        """Send webhook alert."""
        try:
            response = requests.post(
                self.webhook_url,
                json=anomaly_data,
                timeout=10
            )
            response.raise_for_status()
            logger.info("Webhook alert sent successfully")
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def _write_to_file(self, anomaly_data: Dict[str, Any]):
        """Write alert to file."""
        try:
            with open(self.alert_file, 'a') as f:
                f.write(json.dumps(anomaly_data) + '\n')
        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")


class LogFileWatcher(FileSystemEventHandler):
    """Watches log files for changes and queues new entries."""
    
    def __init__(self, log_queue: Queue, target_file: str):
        self.log_queue = log_queue
        self.target_file = Path(target_file)
        self.last_position = 0
        self.line_counter = 0
        
        # Initialize position if file exists
        if self.target_file.exists():
            try:
                self.last_position = self.target_file.stat().st_size
                # Count existing lines
                with open(self.target_file, 'r', encoding='utf-8', errors='ignore') as f:
                    self.line_counter = sum(1 for _ in f)
            except Exception as e:
                logger.error(f"Error initializing file watcher: {e}")
                self.last_position = 0
                self.line_counter = 0
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory or Path(event.src_path) != self.target_file:
            return
        
        self._read_new_lines()
    
    def _read_new_lines(self):
        """Read new lines from the file."""
        try:
            if not self.target_file.exists():
                return
            
            current_size = self.target_file.stat().st_size
            if current_size <= self.last_position:
                return
            
            with open(self.target_file, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(self.last_position)
                new_lines = f.readlines()
                self.last_position = f.tell()
            
            for line in new_lines:
                if line.strip():
                    self.line_counter += 1
                    log_entry = LogEntry(line, self.line_counter)
                    self.log_queue.put(log_entry)
                    
        except Exception as e:
            logger.error(f"Error reading new lines: {e}")


class RealTimeAnomalyDetectionSystem:
    """Main system coordinating all components."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_queue = Queue()
        self.processing_queue = deque()
        self.feature_extractor = FeatureExtractor()
        self.anomaly_detector = AnomalyDetector(
            algorithm=config['algorithm'],
            contamination=config['contamination']
        )
        self.alert_manager = AlertManager(
            webhook_url=config.get('webhook_url'),
            alert_file=config.get('alert_file'),
            use_syslog=config.get('use_syslog', False)
        )
        self.observer = None
        self.is_running = False
        self.batch_size = config.get('batch_size', 100)
        self.batch_timeout = config.get('batch_timeout', 10)
        
        # Set minimum training samples based on batch size
        min_training_samples = max(50, self.batch_size * 2)
        self.anomaly_detector.min_samples_for_training = min_training_samples
        self.output_file = config.get('output_file')
        
        # Load pre-trained model if specified
        if config.get('model_path') and Path(config['model_path']).exists():
            self.anomaly_detector.load_models(config['model_path'])
    
    def start(self):
        """Start the real-time monitoring system."""
        logger.info("Starting Real-time Anomaly Detection System")
        self.is_running = True
        
        # Setup file watcher
        self._setup_file_watcher()
        
        # Start processing thread
        processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        processing_thread.start()
        
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down system...")
            self.stop()
    
    def stop(self):
        """Stop the monitoring system."""
        self.is_running = False
        if self.observer:
            self.observer.stop()
            self.observer.join()
        
        # Process remaining items in queue
        self._process_batch(force=True)
        
        # Save models if specified
        if self.config.get('save_model_path'):
            self.anomaly_detector.save_models(self.config['save_model_path'])
    
    def _setup_file_watcher(self):
        """Setup file system watcher."""
        target_file = Path(self.config['input_file'])
        
        if not target_file.parent.exists():
            target_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file if it doesn't exist
        if not target_file.exists():
            target_file.touch()
        
        event_handler = LogFileWatcher(self.log_queue, str(target_file))
        self.observer = Observer()
        self.observer.schedule(event_handler, str(target_file.parent), recursive=False)
        self.observer.start()
        
        logger.info(f"Watching file: {target_file}")
    
    def _processing_loop(self):
        """Main processing loop."""
        last_batch_time = time.time()
        
        while self.is_running:
            # Check for new log entries
            while not self.log_queue.empty():
                try:
                    log_entry = self.log_queue.get_nowait()
                    self.processing_queue.append(log_entry)
                except:
                    break
            
            # Process batch if conditions are met
            current_time = time.time()
            should_process = (
                len(self.processing_queue) >= self.batch_size or
                (self.processing_queue and 
                 current_time - last_batch_time >= self.batch_timeout)
            )
            
            if should_process:
                self._process_batch()
                last_batch_time = current_time
            
            time.sleep(0.1)  # Small delay to prevent busy waiting
    
    def _process_batch(self, force: bool = False):
        """Process a batch of log entries."""
        if not self.processing_queue and not force:
            return
        
        batch = list(self.processing_queue)
        self.processing_queue.clear()
        
        if not batch:
            return
        
        logger.info(f"Processing batch of {len(batch)} log entries")
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(batch)
            
            if features.shape[0] == 0:
                logger.warning("No features extracted from batch")
                return
            
            # Accumulate training data or predict
            if not self.anomaly_detector.is_fitted:
                training_completed = self.anomaly_detector.accumulate_training_data(features)
                if training_completed:
                    logger.info("🎯 Initial model training completed! Starting anomaly detection...")
                else:
                    total_accumulated = sum(arr.shape[0] for arr in self.anomaly_detector.training_data)
                    logger.info(f"📊 Accumulated {total_accumulated}/{self.anomaly_detector.min_samples_for_training} samples for training")
                return  # Don't predict during training phase
            
            # Predict anomalies on new data
            predictions = self.anomaly_detector.predict(features)
            
            if predictions:
                # Process results
                self._handle_predictions(batch, predictions)
            else:
                logger.info("⏳ No predictions available (models still training)")
            
        except Exception as e:
            logger.error(f"❌ Error processing batch: {e}")
    
    def _handle_predictions(self, batch: List[LogEntry], predictions: Dict[str, Dict]):
        """Handle prediction results and send alerts."""
        anomalies_found = []
        
        for i, log_entry in enumerate(batch):
            for algorithm, results in predictions.items():
                pred = results['predictions'][i]
                score = results['scores'][i]
                
                # Check if anomaly (-1 indicates anomaly in sklearn)
                if pred == -1:
                    anomaly_data = {
                        'log_id': log_entry.log_id,
                        'anomaly_score': float(score),
                        'anomaly_label': algorithm,
                        'raw_line': log_entry.raw_line,
                        'timestamp': log_entry.timestamp,
                        'algorithm': algorithm,
                        'line_number': log_entry.line_number
                    }
                    
                    anomalies_found.append(anomaly_data)
                    self.alert_manager.send_alert(anomaly_data)
        
        # Write anomalies to output file
        if anomalies_found and self.output_file:
            self._write_anomalies_to_file(anomalies_found)
        
        logger.info(f"Found {len(anomalies_found)} anomalies in batch")
    
    def _write_anomalies_to_file(self, anomalies: List[Dict[str, Any]]):
        """Write detected anomalies to output file."""
        try:
            with open(self.output_file, 'a') as f:
                for anomaly in anomalies:
                    f.write(json.dumps(anomaly) + '\n')
        except Exception as e:
            logger.error(f"Failed to write anomalies to output file: {e}")


def create_sample_log_file(file_path: str, num_entries: int = 100):
    """Create a sample JSONL log file for testing."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    sample_logs = []
    for i in range(num_entries):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "message": f"Sample log message {i}",
            "user_id": f"user_{i % 10}",
            "response_time": float(np.random.normal(200, 50)),
            "status_code": 200 if np.random.random() > 0.1 else 500,
            "ip_address": f"192.168.1.{np.random.randint(1, 255)}"
        }
        sample_logs.append(json.dumps(log_entry))
    
    with open(file_path, 'w') as f:
        f.write('\n'.join(sample_logs) + '\n')
    
    logger.info(f"Created sample log file: {file_path}")


def main():
    """Main entry point."""
    colorama_init(autoreset=True)
    parser = argparse.ArgumentParser(
        description="Real-time Log Anomaly Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input logs/app.jsonl --algorithm isolation_forest
  %(prog)s --input logs/app.jsonl --algorithm all --contamination 0.05
  %(prog)s --input logs/app.jsonl --webhook-url http://alerts.company.com/webhook
  %(prog)s --input logs/app.jsonl --model-path trained_model.pkl
        """
    )
    
    # Required arguments
    parser.add_argument('--input', required=True, help='Path to the JSONL log file to monitor')
    
    # Algorithm selection
    parser.add_argument('--algorithm', choices=['isolation_forest', 'lof', 'one_class_svm', 'all'],
                       default='isolation_forest', help='Anomaly detection algorithm(s) to use')
    
    # Model parameters
    parser.add_argument('--contamination', type=float, default=0.1,
                       help='Expected proportion of anomalies (0.0-0.5)')
    
    # I/O options
    parser.add_argument('--output', help='Output file for detected anomalies')
    parser.add_argument('--model-path', help='Path to load pre-trained model')
    parser.add_argument('--save-model', help='Path to save trained model on exit')
    
    # Alerting options
    parser.add_argument('--webhook-url', help='Webhook URL for alerts')
    parser.add_argument('--alert-file', help='File to write alerts to')
    parser.add_argument('--syslog', action='store_true', help='Send alerts to syslog')
    
    # Processing options
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Number of log entries to process in each batch')
    parser.add_argument('--batch-timeout', type=int, default=10,
                       help='Maximum seconds to wait before processing partial batch')
    
    # Utility options
    parser.add_argument('--create-sample', action='store_true',
                       help='Create a sample log file for testing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create sample file if requested
    if args.create_sample:
        create_sample_log_file(args.input, 1000)
        print(f"Sample log file created: {args.input}")
        return
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {args.input}")
        logger.info("Use --create-sample to generate a test file")
        sys.exit(1)
    
    # Create configuration
    config = {
        'input_file': args.input,
        'algorithm': args.algorithm,
        'contamination': args.contamination,
        'output_file': args.output,
        'model_path': args.model_path,
        'save_model_path': args.save_model,
        'webhook_url': args.webhook_url,
        'alert_file': args.alert_file,
        'use_syslog': args.syslog,
        'batch_size': args.batch_size,
        'batch_timeout': args.batch_timeout
    }
    
    # Create and start the system
    system = RealTimeAnomalyDetectionSystem(config)
    
    try:
        system.start()
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
