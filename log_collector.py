#!/usr/bin/env python3
"""
Log receiver server that writes received logs to .log files
and shows its own operational logs
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import socket
import threading
import argparse
import logging
import os
from datetime import datetime

# Configure logging for the server itself
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [SERVER] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class LogReceiver(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/logs':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                source_host = data.get('source', 'unknown')
                logs = data.get('logs', [])
                
                logger.info(f"Received {len(logs)} logs from {source_host}")
                
                # Write logs to file
                self.write_logs_to_file(logs, source_host)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "received"}')
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                self.send_response(400)
                self.end_headers()
                
        else:
            self.send_response(404)
            self.end_headers()
    
    def write_logs_to_file(self, logs, source_host):
        """Write logs to file as they are"""
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"logs_{source_host}_{timestamp}.log"
        
        try:
            with open(filename, 'a', encoding='utf-8') as f:
                for log_entry in logs:
                    # Write the log exactly as received
                    f.write(log_entry + '\n')
            
            logger.info(f"Wrote {len(logs)} logs to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to write logs to file: {e}")
    
    def log_message(self, format, *args):
        # Suppress default HTTP logging to avoid clutter
        pass

class TCPLogReceiver:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.clients = []
        
    def start(self):
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            logger.info(f"TCP log receiver listening on {self.host}:{self.port}")
            
            while True:
                client_socket, address = self.socket.accept()
                client_host = address[0]
                logger.info(f"New TCP connection from {client_host}:{address[1]}")
                
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, client_host),
                    daemon=True
                )
                client_thread.start()
                
        except Exception as e:
            logger.error(f"TCP server error: {e}")
    
    def handle_client(self, client_socket, client_host):
        buffer = ""
        log_count = 0
        
        # Create log file for this client
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"logs_{client_host}_{timestamp}.log"
        
        try:
            with open(filename, 'a', encoding='utf-8') as log_file:
                while True:
                    try:
                        data = client_socket.recv(4096).decode('utf-8')
                        if not data:
                            break
                            
                        buffer += data
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            if line.strip():
                                # Write the log exactly as received
                                log_file.write(line.strip() + '\n')
                                log_file.flush()  # Ensure immediate write
                                log_count += 1
                                
                                # Log every 10 received logs
                                if log_count % 10 == 0:
                                    logger.info(f"Received {log_count} logs from {client_host}")
                                
                    except ConnectionResetError:
                        break
                    except UnicodeDecodeError as e:
                        logger.warning(f"Unicode decode error from {client_host}: {e}")
                        
        except Exception as e:
            logger.error(f"Error handling client {client_host}: {e}")
        finally:
            client_socket.close()
            logger.info(f"Client {client_host} disconnected. Total logs received: {log_count}")

class UDPLogReceiver:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client_files = {}  # Track open files per client
        self.client_counts = {}  # Track log counts per client
        
    def start(self):
        try:
            self.socket.bind((self.host, self.port))
            logger.info(f"UDP log receiver listening on {self.host}:{self.port}")
            
            while True:
                try:
                    data, address = self.socket.recvfrom(4096)
                    client_host = address[0]
                    message = data.decode('utf-8').strip()
                    
                    if message:
                        self.write_udp_log(message, client_host)
                        
                except Exception as e:
                    logger.error(f"Error receiving UDP: {e}")
                    
        except Exception as e:
            logger.error(f"UDP server error: {e}")
    
    def write_udp_log(self, message, client_host):
        """Write UDP log to file"""
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"logs_{client_host}_{timestamp}.log"
        
        try:
            # Track client log count
            if client_host not in self.client_counts:
                self.client_counts[client_host] = 0
                logger.info(f"New UDP client: {client_host}")
            
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
                
            self.client_counts[client_host] += 1
            
            # Log every 10 received logs per client
            if self.client_counts[client_host] % 10 == 0:
                logger.info(f"Received {self.client_counts[client_host]} logs from {client_host}")
                
        except Exception as e:
            logger.error(f"Failed to write UDP log from {client_host}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Log receiver server that writes logs to files')
    parser.add_argument('-p', '--port', type=int, default=8080, help='Port to listen on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--protocol', choices=['http', 'tcp', 'udp'], default='http',
                       help='Protocol to use')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose server logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure log directory exists
    os.makedirs('logs', exist_ok=True)
    os.chdir('logs')  # Change to logs directory
    
    logger.info(f"Starting {args.protocol.upper()} log receiver server")
    logger.info(f"Listening on {args.host}:{args.port}")
    logger.info(f"Logs will be written to: {os.getcwd()}")
    logger.info("Press Ctrl+C to stop")
    
    try:
        if args.protocol == 'http':
            server = HTTPServer((args.host, args.port), LogReceiver)
            logger.info("HTTP server ready to receive logs at /logs endpoint")
            server.serve_forever()
            
        elif args.protocol == 'tcp':
            receiver = TCPLogReceiver(args.host, args.port)
            receiver.start()
            
        elif args.protocol == 'udp':
            receiver = UDPLogReceiver(args.host, args.port)
            receiver.start()
            
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    main()
