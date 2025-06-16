#!/usr/bin/env python3
import os
import time
import json
import re
from datetime import datetime

# Configuration constants
INPUT_FILE = "logs/logs_fedora_20250606.log"
OUTPUT_FILE = "logs/logs_parsed.json"
POLL_INTERVAL = 0.1  # seconds between file checks
FILE_CHECK_INTERVAL = 1.0  # seconds between file existence checks

def wait_for_file(filepath):
    """
    Wait for the input file to exist before proceeding.
    Checks every FILE_CHECK_INTERVAL seconds until file is found.
    """
    while not os.path.exists(filepath):
        print(f"Waiting for file {filepath} to be created...")
        time.sleep(FILE_CHECK_INTERVAL)
    print(f"File {filepath} found, starting to monitor...")

def parse_systemd_log_line(line):
    """
    Parse a single systemd journal log line into structured JSON format.
    Expected format: MMM DD HH:MM:SS hostname service[pid]: message
    Returns a dictionary with parsed fields or None if parsing fails.
    """
    # Remove trailing newline and whitespace
    line = line.strip()
    if not line:
        return None
    
    # Regex pattern to match systemd journal format
    # Pattern: Month Day Time Hostname Service[PID]: Message
    pattern = r'^(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+([^[\s]+)(?:\[(\d+)\])?\s*:\s*(.*)$'
    
    match = re.match(pattern, line)
    if not match:
        # If standard format doesn't match, try alternative patterns
        # Pattern for logs without PID: Month Day Time Hostname Service: Message
        alt_pattern = r'^(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+([^:\s]+)\s*:\s*(.*)$'
        alt_match = re.match(alt_pattern, line)
        
        if alt_match:
            timestamp_str, hostname, service, message = alt_match.groups()
            pid = None
        else:
            # If no pattern matches, create a generic entry
            return {
                "timestamp": datetime.now().isoformat(),
                "hostname": "unknown",
                "service": "unknown",
                "pid": None,
                "message": line,
                "raw_line": line
            }
    else:
        timestamp_str, hostname, service, pid, message = match.groups()
    
    # Convert timestamp to ISO format (add current year since syslog doesn't include year)
    try:
        current_year = datetime.now().year
        full_timestamp = f"{current_year} {timestamp_str}"
        parsed_time = datetime.strptime(full_timestamp, "%Y %b %d %H:%M:%S")
        iso_timestamp = parsed_time.isoformat()
    except ValueError:
        # If timestamp parsing fails, use current time
        iso_timestamp = datetime.now().isoformat()
    
    # Create structured log entry
    log_entry = {
        "timestamp": iso_timestamp,
        "hostname": hostname,
        "service": service,
        "pid": int(pid) if pid and pid.isdigit() else None,
        "message": message.strip() if 'message' in locals() else "",
        "raw_line": line
    }
    
    return log_entry

def append_json_to_file(json_obj, output_file):
    """
    Append a JSON object as a single line to the output file.
    Each JSON object is written on its own line for easy parsing.
    """
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            json.dump(json_obj, f, ensure_ascii=False, separators=(',', ':'))
            f.write('\n')
            f.flush()  # Ensure data is written immediately
    except Exception as e:
        print(f"Error writing to output file: {e}")

def tail_file(filepath):
    """
    Generator function that yields new lines from a file as they are appended.
    Maintains file position to avoid re-reading existing content.
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        # Move to end of file to start reading only new content
        f.seek(0, 2)  # Seek to end of file
        
        while True:
            line = f.readline()
            if line:
                yield line
            else:
                # No new line available, wait before checking again
                time.sleep(POLL_INTERVAL)

def initialize_output_file(output_file):
    """
    Initialize the output file if it doesn't exist.
    Creates an empty file ready for JSON line appending.
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

def main():
    """
    Main function that orchestrates the log parsing process.
    Handles file monitoring, parsing, and output generation.
    """
    print("Starting systemd journal log parser...")
    
    # Wait for input file to exist
    wait_for_file(INPUT_FILE)
    
    # Initialize output file
    if not initialize_output_file(OUTPUT_FILE):
        print("Failed to initialize output file. Exiting.")
        return
    
    print(f"Monitoring {INPUT_FILE} for new log entries...")
    print(f"Parsed logs will be written to {OUTPUT_FILE}")
    
    try:
        # Start tailing the input file
        line_count = 0
        for line in tail_file(INPUT_FILE):
            # Parse the log line
            parsed_entry = parse_systemd_log_line(line)
            
            if parsed_entry:
                # Write parsed entry to output file
                append_json_to_file(parsed_entry, OUTPUT_FILE)
                line_count += 1
                
                # Optional: Print progress every 100 lines
                if line_count % 100 == 0:
                    print(f"Processed {line_count} log entries...")
            
    except KeyboardInterrupt:
        print(f"\nLog parser stopped. Processed {line_count} total entries.")
    except Exception as e:
        print(f"Error during log processing: {e}")

if __name__ == "__main__":
    main()
