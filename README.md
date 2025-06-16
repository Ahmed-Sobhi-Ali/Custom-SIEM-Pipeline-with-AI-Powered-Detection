# Building a Full SIEM Pipeline from Scratch with Real-Time AI-Based Anomaly Detection

## Introduction

Security Information and Event Management (SIEM) systems are central to any modern cybersecurity infrastructure. They aggregate logs, parse events, extract features, and detect abnormal behavior in real time. While many commercial solutions exist, this project demonstrates how to build a full SIEM pipeline from scratch using Python, with a modular design and real-time AI-based anomaly detection.

This article explains every component in detail—from kernel-level log collection to machine learning-based threat detection—backed by live monitoring, alerting mechanisms, and extensible architecture.

---

## Project Architecture Overview

The system is composed of the following main components:

1. **Log Forwarder**: Captures real-time logs from `systemd.journal`.
2. **Log Receiver**: Receives logs over HTTP, TCP, or UDP and saves them to raw log files.
3. **Log Parser**: Converts raw logs into structured JSON.
4. **Feature Extractor**: Enhances logs with engineered features.
5. **AI-Based Anomaly Detector**: Detects suspicious patterns using ML models.
6. **Alert Manager**: Sends alerts via Webhook, Syslog, file, or console.
7. **File Watcher**: Monitors parsed logs and triggers batch processing.

Each component operates independently but integrates through shared interfaces and real-time pipelines.

---

## 1. Log Forwarder (Kernel-Level Collection)

* **File**: `log_forwarder.py`
* **Source**: `systemd.journal` via `systemd` Python bindings
* **Protocols Supported**: HTTP, TCP, UDP, and Syslog
* **Features**:

  * Filters by systemd unit
  * Batching and exponential backoff retry logic
  * Formats logs similar to `journalctl`

This component reads directly from the system's journal buffer, enabling zero-agent, zero-delay log forwarding. Logs are batched and sent periodically to the receiver.

---

## 2. Log Receiver

* **File**: `log_collector.py`
* **Supported Modes**: HTTP server (`/logs` endpoint), TCP server, UDP listener
* **Storage**: Stores logs per source host with date-based naming
* **Concurrency**: Multithreaded TCP handling for multiple clients

The receiver normalizes all incoming logs, appends them to `.log` files, and prepares them for the parser. This enables deployment in distributed environments or agent-based log sources.

---

## 3. Log Parser

* **File**: `parser.py`
* **Input**: Raw `.log` files
* **Output**: Structured JSON logs in `logs_parsed.json`
* **Parsing Logic**:

  * Regex-based parsing of syslog-style lines
  * Fallback for malformed lines
  * Timestamp normalization to ISO 8601 format

This parser continuously tails the raw logs and converts every line into a structured format that can be enriched and analyzed.

---

## 4. Feature Extractor

* **File**: `features_extraction.py`
* **Input**: `logs_parsed.json`
* **Output**: `logs_features.jsonl`
* **Features Extracted**:

  * **Timestamp**: hour of day, day of week, weekend indicator
  * **Service Info**: name, system/non-system classification
  * **Message Content**: length, presence of error/warning keywords
  * **PID Category**: low/system/user ranges

This enrichment process transforms raw log lines into ML-ready features while preserving the original context.

---

## 5. AI-Based Anomaly Detector

* **Algorithms Supported**: Isolation Forest, LOF, One-Class SVM
* **Modes**:

  * Accumulate data before initial training
  * Periodic batch predictions
* **Feature Handling**:

  * Numerical: standard scaling
  * Text: TF-IDF vectorization
  * Combined: Concatenated vectors
* **File**: `anomaly_detector.py`

Each batch of logs is passed through a feature extractor, then classified by trained anomaly detection models. The system can be preloaded with saved models or trained live.

---

## 6. Alert Manager

* **Alert Channels**:

  * Console (always enabled)
  * Webhook (JSON POST request)
  * Syslog
  * Local file (JSONL format)
* **Alert Format**: Contains log ID, anomaly score, algorithm used, and raw line

This allows flexible deployment in environments with SIEM dashboards, SOAR platforms, or simple log rotation.

---

## 7. Real-Time Log Watcher and Controller

* **Component**: `RealTimeAnomalyDetectionSystem`
* **Design**:

  * Uses `watchdog` to monitor feature log file
  * Collects logs into batches
  * Triggers anomaly detection and alerting
  * Saves model to disk (optional)
* **Batch Controls**:

  * Size-based or timeout-based triggering

This class ties all parts together and ensures that logs are continuously processed, classified, and handled accordingly.

---

## CLI and Deployment

* **Main Entrypoint**: `main()`
* **CLI Options**:

  * `--input`, `--output`
  * `--algorithm`, `--contamination`
  * `--webhook-url`, `--syslog`, `--alert-file`
  * `--model-path`, `--save-model`
  * `--batch-size`, `--batch-timeout`
  * `--create-sample`

You can deploy the entire system with:

```bash
python anomaly_detector.py --input logs/logs_features.jsonl --algorithm all --webhook-url http://localhost:5000/alert
```

Or generate test data using:

```bash
python anomaly_detector.py --input logs/test.jsonl --create-sample
```

---

## Skills Demonstrated

This project demonstrates a wide range of practical skills:

* Low-level system log access via `systemd`
* Multi-protocol network programming (HTTP/TCP/UDP/Syslog)
* Real-time file monitoring and queuing
* Regex-based parsing and structured log normalization
* Feature engineering for temporal, categorical, and text data
* Unsupervised anomaly detection with scikit-learn
* Alerting and integration with external systems
* Scalable architecture with modularity and clean separation

---

## Conclusion

Building a SIEM pipeline from scratch provides deep insight into the internal operations of security systems. This project showcases a fully working, real-time, AI-powered log monitoring and threat detection system designed with production-readiness and modularity in mind.

Whether used as a blueprint for enterprise integration, a training lab, or a research prototype, this project serves as a powerful demonstration of what's possible with Python and disciplined system design.

---

**Author**: Ahmed Sobhi Ali
---

> "Logs don't lie. But only the right systems can hear what they say."
