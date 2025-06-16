#!/usr/bin/env python3
"""
Real-time systemd journal log forwarder
Supports multiple protocols: HTTP, TCP, UDP, Syslog
"""

import json
import socket
import time
import signal
import sys
import threading
from datetime import datetime
from systemd import journal
import requests
from urllib.parse import urljoin
import logging
from queue import Queue, Empty
import argparse

# Configure logging for the forwarder itself
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class LogForwarder:
    def __init__(self, server_host, server_port, protocol='http', batch_size=1, 
                 batch_timeout=5.0, max_retries=3, filter_unit=None):
        self.server_host = server_host
        self.server_port = server_port
        self.protocol = protocol.lower()
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_retries = max_retries
        self.filter_unit = filter_unit
        
        self.running = True
        self.log_queue = Queue()
        self.batch_buffer = []
        self.last_batch_time = time.time()
        
        # Setup journal reader
        self.journal = journal.Reader()
        if filter_unit:
            self.journal.add_match(_SYSTEMD_UNIT=filter_unit)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Initialize connection based on protocol
        self.init_connection()
        
    def signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
    def init_connection(self):
        """Initialize connection based on protocol"""
        if self.protocol == 'tcp':
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            try:
                self.socket.connect((self.server_host, self.server_port))
                logger.info(f"Connected to TCP server {self.server_host}:{self.server_port}")
            except ConnectionRefusedError:
                logger.error(f"Cannot connect to TCP server {self.server_host}:{self.server_port}")
                sys.exit(1)
                
        elif self.protocol == 'udp':
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            logger.info(f"UDP forwarder configured for {self.server_host}:{self.server_port}")
            
        elif self.protocol == 'syslog':
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            logger.info(f"Syslog forwarder configured for {self.server_host}:{self.server_port}")
            
        elif self.protocol == 'http':
            self.session = requests.Session()
            self.session.timeout = 10
            self.endpoint = f"http://{self.server_host}:{self.server_port}/logs"
            logger.info(f"HTTP forwarder configured for {self.endpoint}")
            
    def format_log_entry(self, entry):
        """Format journal entry exactly like journalctl"""
        timestamp = entry.get('__REALTIME_TIMESTAMP', datetime.now())
        if hasattr(timestamp, 'strftime'):
            time_str = timestamp.strftime('%b %d %H:%M:%S')
        else:
            time_str = datetime.now().strftime('%b %d %H:%M:%S')
        
        # Get hostname (like journalctl shows)
        hostname = socket.gethostname()
        
        # Get process info - prefer SYSLOG_IDENTIFIER over _COMM
        process_name = (entry.get('SYSLOG_IDENTIFIER') or 
                       entry.get('_COMM') or 
                       'unknown')
        
        # Get PID
        pid = entry.get('_PID', '')
        if pid:
            process_info = f"{process_name}[{pid}]"
        else:
            process_info = process_name
        
        # Get the actual message
        message = entry.get('MESSAGE', '')
        
        # Format exactly like journalctl: "MMM DD HH:MM:SS hostname process[pid]: message"
        journalctl_format = f"{time_str} {hostname} {process_info}: {message}"
        
        return journalctl_format
    
    def format_syslog_message(self, entry):
        """Format entry as RFC3164 syslog message"""
        timestamp = entry.get('__REALTIME_TIMESTAMP', datetime.now())
        if hasattr(timestamp, 'strftime'):
            time_str = timestamp.strftime('%b %d %H:%M:%S')
        else:
            time_str = datetime.now().strftime('%b %d %H:%M:%S')
            
        hostname = socket.gethostname()
        tag = entry.get('SYSLOG_IDENTIFIER', entry.get('_COMM', 'journal'))
        pid = entry.get('_PID')
        
        if pid:
            tag_pid = f"{tag}[{pid}]"
        else:
            tag_pid = tag
            
        priority = entry.get('PRIORITY', 6)
        facility = entry.get('SYSLOG_FACILITY', 16)  # local0 = 16
        
        # Calculate PRI value: facility * 8 + severity
        pri = facility * 8 + int(priority)
        
        message = entry.get('MESSAGE', '')
        
        return f"<{pri}>{time_str} {hostname} {tag_pid}: {message}"
    
    def send_http_batch(self, entries):
        """Send logs via HTTP POST"""
        payload = {
            'logs': entries,
            'source': socket.gethostname(),
            'timestamp': datetime.now().isoformat()
        }
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    self.endpoint,
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                )
                response.raise_for_status()
                logger.info(f"Successfully sent batch of {len(entries)} logs to {self.server_host}:{self.server_port}")
                return True
                
            except requests.RequestException as e:
                logger.warning(f"HTTP send attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
        logger.error(f"Failed to send {len(entries)} logs after {self.max_retries} attempts")
        return False
    
    def send_tcp_batch(self, entries):
        """Send logs via TCP socket"""
        try:
            for entry in entries:
                message = entry + '\n'  # entry is already formatted string
                self.socket.sendall(message.encode('utf-8'))
            logger.info(f"Successfully sent batch of {len(entries)} logs via TCP to {self.server_host}:{self.server_port}")
            return True
        except (socket.error, BrokenPipeError) as e:
            logger.error(f"TCP send failed: {e}")
            # Try to reconnect
            try:
                self.socket.close()
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(10)
                self.socket.connect((self.server_host, self.server_port))
                logger.info("Reconnected to TCP server")
            except Exception as reconnect_error:
                logger.error(f"Failed to reconnect: {reconnect_error}")
            return False
    
    def send_udp_batch(self, entries):
        """Send logs via UDP socket"""
        try:
            for entry in entries:
                if self.protocol == 'syslog':
                    # For syslog, use the original entry format
                    original_entry = entry.get('_original', {})
                    message = self.format_syslog_message(original_entry) if original_entry else entry
                else:
                    message = entry  # entry is already formatted string
                    
                self.socket.sendto(
                    message.encode('utf-8'),
                    (self.server_host, self.server_port)
                )
            logger.info(f"Successfully sent batch of {len(entries)} logs via {'syslog' if self.protocol == 'syslog' else 'UDP'} to {self.server_host}:{self.server_port}")
            return True
        except socket.error as e:
            logger.error(f"UDP send failed: {e}")
            return False
    
    def send_batch(self, entries):
        """Send batch of log entries based on protocol"""
        if not entries:
            return
            
        if self.protocol == 'http':
            return self.send_http_batch(entries)
        elif self.protocol == 'tcp':
            return self.send_tcp_batch(entries)
        elif self.protocol in ['udp', 'syslog']:
            return self.send_udp_batch(entries)
    
    def batch_sender(self):
        """Background thread to send batched logs"""
        while self.running:
            try:
                # Get log from queue with timeout
                try:
                    log_entry = self.log_queue.get(timeout=1.0)
                    self.batch_buffer.append(log_entry)
                except Empty:
                    pass
                
                current_time = time.time()
                
                # Send batch if buffer is full or timeout reached
                if (len(self.batch_buffer) >= self.batch_size or 
                    (self.batch_buffer and current_time - self.last_batch_time >= self.batch_timeout)):
                    
                    if self.batch_buffer:
                        self.send_batch(self.batch_buffer.copy())
                        self.batch_buffer.clear()
                        self.last_batch_time = current_time
                        
            except Exception as e:
                logger.error(f"Error in batch sender: {e}")
                time.sleep(1)
    
    def start_forwarding(self):
        """Start the log forwarding process"""
        logger.info(f"Starting log forwarding to {self.server_host}:{self.server_port} via {self.protocol.upper()}")
        if self.filter_unit:
            logger.info(f"Filtering logs for unit: {self.filter_unit}")
        
        # Start batch sender thread
        sender_thread = threading.Thread(target=self.batch_sender, daemon=True)
        sender_thread.start()
        
        # Position journal at the end for real-time monitoring
        self.journal.seek_tail()
        self.journal.get_previous()
        
        try:
            while self.running:
                if self.journal.wait(1000) == journal.APPEND:
                    for entry in self.journal:
                        if not self.running:
                            break
                            
                        formatted_entry = self.format_log_entry(entry)
                        
                        # For syslog, keep original entry for proper formatting
                        if self.protocol == 'syslog':
                            log_data = {
                                'formatted': formatted_entry,
                                '_original': entry
                            }
                        else:
                            log_data = formatted_entry
                            
                        # Add to queue for batch processing
                        self.log_queue.put(log_data)
                        
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        self.running = False
        
        # Send any remaining logs
        if self.batch_buffer:
            self.send_batch(self.batch_buffer)
            
        # Close connections
        if hasattr(self, 'socket'):
            self.socket.close()
        if hasattr(self, 'session'):
            self.session.close()
            
        self.journal.close()
        logger.info("Cleanup completed")

def main():
    parser = argparse.ArgumentParser(description='Forward systemd journal logs to remote server')
    parser.add_argument('server', help='Server hostname or IP')
    parser.add_argument('port', type=int, help='Server port')
    parser.add_argument('-p', '--protocol', choices=['http', 'tcp', 'udp', 'syslog'], 
                       default='http', help='Protocol to use (default: http)')
    parser.add_argument('-b', '--batch-size', type=int, default=10, 
                       help='Number of logs to batch before sending (default: 10)')
    parser.add_argument('-t', '--timeout', type=float, default=5.0,
                       help='Batch timeout in seconds (default: 5.0)')
    parser.add_argument('-r', '--retries', type=int, default=3,
                       help='Max retry attempts for failed sends (default: 3)')
    parser.add_argument('-u', '--unit', help='Filter logs by systemd unit')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        forwarder = LogForwarder(
            server_host=args.server,
            server_port=args.port,
            protocol=args.protocol,
            batch_size=args.batch_size,
            batch_timeout=args.timeout,
            max_retries=args.retries,
            filter_unit=args.unit
        )
        
        forwarder.start_forwarding()
        
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()