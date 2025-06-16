#!/usr/bin/env python3
import json
import re
import os
import time
from datetime import datetime

# Configuration constants
INPUT_FILE = "logs/logs_parsed.json"
OUTPUT_FILE = "logs/logs_features.jsonl"
POLL_INTERVAL = 0.1  # seconds between file checks
FILE_CHECK_INTERVAL = 1.0  # seconds between file existence checks

# Predefined lists for feature extraction
SYSTEM_SERVICES = {
    'systemd', 'kernel', 'gnome-shell', 'NetworkManager', 'sshd', 'chronyd',
    'dbus', 'avahi-daemon', 'cups', 'gdm', 'pulseaudio', 'bluetoothd',
    'firewalld', 'accounts-daemon', 'packagekit', 'udisks2', 'polkitd',
    'rtkit-daemon', 'colord', 'ModemManager', 'wpa_supplicant', 'dhclient',
    'systemd-logind', 'systemd-resolved', 'systemd-timesyncd', 'systemd-networkd'
}

ERROR_KEYWORDS = [
    'keysym', 'xkbcomp', 'xf86', 'failed', 'not found', 'timeout', 'denied',
    'refused', 'rejected', 'invalid', 'corrupt', 'missing', 'unavailable',
    'unreachable', 'broken', 'damaged', 'fatal', 'critical', 'abort',
    'segfault', 'panic', 'exception', 'overflow', 'underflow'
]

def wait_for_file(filepath):
    """
    Wait for the input file to exist before proceeding.
    Checks every FILE_CHECK_INTERVAL seconds until file is found.
    """
    while not os.path.exists(filepath):
        print(f"Waiting for file {filepath} to be created...")
        time.sleep(FILE_CHECK_INTERVAL)
    print(f"File {filepath} found, starting to monitor...")

def tail_json_file(filepath):
    """
    Generator function that yields new JSON lines from a file as they are appended.
    Maintains file position to avoid re-reading existing content.
    Handles JSON parsing errors gracefully.
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        # Move to end of file to start reading only new content
        f.seek(0, 2)  # Seek to end of file
        
        line_num = 0
        while True:
            line = f.readline()
            if line:
                line_num += 1
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        json_obj = json.loads(line)
                        yield json_obj, line_num
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON on line {line_num}: {e}")
                        continue
            else:
                # No new line available, wait before checking again
                time.sleep(POLL_INTERVAL)

def append_json_to_output(json_obj, output_file):
    """
    Append a JSON object as a single line to the output file.
    Each JSON object is written on its own line for JSONL format.
    """
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            json.dump(json_obj, f, ensure_ascii=False, separators=(',', ':'))
            f.write('\n')
            f.flush()  # Ensure data is written immediately
    except Exception as e:
        print(f"Error writing to output file: {e}")

def initialize_output_file(output_file):
    """
    Initialize the output file if it doesn't exist.
    Creates an empty file ready for JSONL appending.
    """
    if not os.path.exists(output_file):
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                pass  # Create empty file
            print(f"Created output file: {output_file}")
        except Exception as e:
            print(f"Error creating output file: {e}")
            return False
    return True

def extract_timestamp_features(timestamp_str):
    """
    Extract time-based features from ISO timestamp string.
    Returns dictionary with hour_of_day, day_of_week, and is_weekend.
    """
    try:
        # Parse ISO timestamp
        dt = datetime.fromisoformat(timestamp_str)
        
        # Extract hour (0-23)
        hour_of_day = dt.hour
        
        # Extract day of week (Monday=0, Sunday=6)
        weekday = dt.weekday()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                    'Friday', 'Saturday', 'Sunday']
        day_of_week = day_names[weekday]
        
        # Check if weekend (Saturday=5, Sunday=6)
        is_weekend = weekday >= 5
        
        return {
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend
        }
    except (ValueError, AttributeError) as e:
        # Return default values if timestamp parsing fails
        return {
            'hour_of_day': 0,
            'day_of_week': 'Unknown',
            'is_weekend': False
        }

def extract_service_features(service_name):
    """
    Extract service-related features.
    Returns dictionary with service name and system service classification.
    """
    if not service_name:
        service_name = "unknown"
    
    # Check if service is in predefined system services list
    is_system_service = service_name.lower() in SYSTEM_SERVICES
    
    return {
        'service': service_name,
        'is_system_service': is_system_service
    }

def extract_message_features(message):
    """
    Extract message content features including length, warnings, errors, and keywords.
    Returns dictionary with message analysis results.
    """
    if not message:
        message = ""
    
    # Calculate message length
    msg_length = len(message)
    
    # Check for warning indicators (case-insensitive)
    has_warning = bool(re.search(r'\bwarning\b', message, re.IGNORECASE))
    
    # Check for error indicators (case-insensitive)
    has_error = bool(re.search(r'\berror\b', message, re.IGNORECASE))
    
    # Find matching error keywords (case-insensitive)
    message_lower = message.lower()
    matched_keywords = []
    
    for keyword in ERROR_KEYWORDS:
        if keyword.lower() in message_lower:
            matched_keywords.append(keyword)
    
    return {
        'msg_length': msg_length,
        'has_warning': has_warning,
        'has_error': has_error,
        'error_keywords': matched_keywords
    }

def extract_pid_features(pid):
    """
    Extract PID-based features to classify process range.
    Returns dictionary with pid_range classification.
    """
    if pid is None or not isinstance(pid, (int, str)):
        return {'pid_range': 'unknown'}
    
    try:
        pid_num = int(pid)
        
        if pid_num < 1000:
            pid_range = 'low'
        elif 1000 <= pid_num < 10000:
            pid_range = 'system'
        else:
            pid_range = 'user'
            
        return {'pid_range': pid_range}
    except (ValueError, TypeError):
        return {'pid_range': 'unknown'}

def process_log_entry(log_entry, log_id):
    """
    Process a single log entry and extract all features.
    Returns a dictionary containing original data plus extracted features.
    """
    # Start with the original log entry
    result = log_entry.copy()
    
    # Add log identifier
    result['log_id'] = log_id
    
    # Extract timestamp features
    timestamp_features = extract_timestamp_features(log_entry.get('timestamp', ''))
    result.update(timestamp_features)
    
    # Extract service features
    service_features = extract_service_features(log_entry.get('service', ''))
    result.update(service_features)
    
    # Extract message features
    message_features = extract_message_features(log_entry.get('message', ''))
    result.update(message_features)
    
    # Extract PID features
    pid_features = extract_pid_features(log_entry.get('pid'))
    result.update(pid_features)
    
    return result

def process_log_file_realtime(input_file, output_file):
    """
    Process the input JSON log file in real-time, tailing new entries as they arrive.
    Continuously monitors the file and processes new JSON lines as they are appended.
    """
    processed_count = 0
    error_count = 0
    
    print(f"Starting real-time processing of {input_file}...")
    print(f"Enriched logs will be written to {output_file}")
    print("Press Ctrl+C to stop processing...")
    
    try:
        # Start tailing the input file
        for log_entry, line_num in tail_json_file(input_file):
            try:
                # Process the log entry and extract features
                enriched_entry = process_log_entry(log_entry, line_num)
                
                # Write enriched entry to output file
                append_json_to_output(enriched_entry, output_file)
                
                processed_count += 1
                
                # Progress indicator every 100 entries
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} log entries...")
                    
            except Exception as e:
                print(f"Error processing log entry on line {line_num}: {e}")
                error_count += 1
                continue
                
    except KeyboardInterrupt:
        print(f"\nReal-time processing stopped by user.")
        print(f"Total processed: {processed_count} entries")
        print(f"Total errors: {error_count} entries")
    except Exception as e:
        print(f"Error during real-time processing: {e}")
        return False
    
    return True

def main():
    """
    Main function that orchestrates the real-time log feature extraction process.
    """
    print("Starting real-time JSON log feature extraction...")
    
    # Wait for input file to exist
    wait_for_file(INPUT_FILE)
    
    # Initialize output file
    if not initialize_output_file(OUTPUT_FILE):
        print("Failed to initialize output file. Exiting.")
        return
    
    # Process the log file in real-time
    success = process_log_file_realtime(INPUT_FILE, OUTPUT_FILE)
    
    if success:
        print("Real-time feature extraction completed!")
    else:
        print("Real-time feature extraction failed.")

if __name__ == "__main__":
    main()
