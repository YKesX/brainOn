#!/usr/bin/env python3
"""
Main Real-Time Data Processing Pipeline
Implements Task 17 from tasks_final.yml

Multi-threaded Architecture:
- SerialThread: Handles ESP32 communication  
- ProcessingThread: EEG classification pipeline
- WebSocketThread: Broadcasts results
- MainThread: Coordination and UI

Features:
- Real-time EEG feature extraction and classification
- Muscle state monitoring
- WebSocket broadcasting for crypto wallet integration
- Data logging and debugging
- Performance monitoring
"""

import argparse
import logging
import time
import threading
import queue
import signal
import sys
import json
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
from collections import deque

# Import our custom modules
from host.serial_manager import SerialManager, MuscleData, EEGData, SystemStatus
from eeg_classifier import EEGClassifier

class Config:
    """Configuration settings for the real-time system"""
    
    # Serial communication
    SERIAL_PORT = None  # Auto-detect
    SERIAL_BAUDRATE = 115200
    
    # EEG processing
    EEG_WINDOW_SIZE = 128  # Samples for feature extraction (1.28s at 100Hz)
    EEG_STEP_SIZE = 64    # 50% overlap
    EEG_SAMPLE_RATE = 100  # Hz
    
    # Classification
    MODEL_PATH = 'models/rf_eeg_classifier.pkl'
    CONFIDENCE_THRESHOLD = 0.7
    SMOOTHING_WINDOW = 5
    
    # WebSocket
    WEBSOCKET_HOST = 'localhost'
    WEBSOCKET_PORT = 5000
    
    # Data logging
    LOG_MUSCLE_DATA = True
    LOG_EEG_DATA = True
    LOG_CLASSIFICATIONS = True
    DATA_LOG_DIR = 'logs'
    
    # Performance
    MAX_QUEUE_SIZE = 1000
    PERFORMANCE_LOG_INTERVAL = 10  # seconds

class PerformanceMonitor:
    """Monitor system performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'start_time': time.time(),
            'eeg_samples_processed': 0,
            'muscle_samples_processed': 0,
            'classifications_made': 0,
            'websocket_messages_sent': 0,
            'errors': 0,
            'avg_classification_latency': 0.0,
            'last_eeg_timestamp': 0,
            'last_muscle_timestamp': 0,
            'data_gaps': 0
        }
        self.latency_history = deque(maxlen=100)
        
    def record_eeg_sample(self, timestamp):
        self.metrics['eeg_samples_processed'] += 1
        self.metrics['last_eeg_timestamp'] = timestamp
        
    def record_muscle_sample(self, timestamp):
        self.metrics['muscle_samples_processed'] += 1
        self.metrics['last_muscle_timestamp'] = timestamp
        
    def record_classification(self, latency_ms):
        self.metrics['classifications_made'] += 1
        self.latency_history.append(latency_ms)
        self.metrics['avg_classification_latency'] = np.mean(list(self.latency_history))
        
    def record_websocket_send(self):
        self.metrics['websocket_messages_sent'] += 1
        
    def record_error(self):
        self.metrics['errors'] += 1
        
    def record_data_gap(self):
        self.metrics['data_gaps'] += 1
        
    def get_metrics(self):
        uptime = time.time() - self.metrics['start_time']
        metrics = self.metrics.copy()
        metrics['uptime'] = uptime
        metrics['eeg_rate'] = metrics['eeg_samples_processed'] / uptime if uptime > 0 else 0
        metrics['muscle_rate'] = metrics['muscle_samples_processed'] / uptime if uptime > 0 else 0
        metrics['classification_rate'] = metrics['classifications_made'] / uptime if uptime > 0 else 0
        return metrics

class DataLogger:
    """Handle data logging to files"""
    
    def __init__(self, log_dir='logs', enabled=True):
        self.log_dir = log_dir
        self.enabled = enabled
        
        if enabled:
            import os
            os.makedirs(log_dir, exist_ok=True)
            
            # Create log files with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.muscle_file = open(f'{log_dir}/muscle_data_{timestamp}.csv', 'w')
            self.eeg_file = open(f'{log_dir}/eeg_data_{timestamp}.csv', 'w')
            self.classification_file = open(f'{log_dir}/classifications_{timestamp}.csv', 'w')
            
            # Write headers
            self.muscle_file.write('timestamp,binary,decimal,raw_line\n')
            self.eeg_file.write('timestamp,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10,ch11,ch12,esp32_timestamp\n')
            self.classification_file.write('timestamp,class,class_id,confidence,processing_time,smoothed,probabilities\n')
    
    def log_muscle_data(self, muscle_data: MuscleData):
        if self.enabled and Config.LOG_MUSCLE_DATA:
            self.muscle_file.write(f'{time.time()},{muscle_data.binary},{muscle_data.decimal},{muscle_data.raw_line}\n')
            self.muscle_file.flush()
    
    def log_eeg_data(self, eeg_data: EEGData):
        if self.enabled and Config.LOG_EEG_DATA:
            channels_str = ','.join(map(str, eeg_data.channels))
            self.eeg_file.write(f'{time.time()},{channels_str},{eeg_data.timestamp}\n')
            self.eeg_file.flush()
    
    def log_classification(self, result: Dict[str, Any]):
        if self.enabled and Config.LOG_CLASSIFICATIONS:
            probs_str = '|'.join(map(str, result.get('probabilities', [])))
            self.classification_file.write(
                f'{time.time()},{result.get("class","")},{result.get("class_id","")}'
                f',{result.get("confidence",0)},{result.get("processing_time",0)}'
                f',{result.get("smoothed",False)},{probs_str}\n'
            )
            self.classification_file.flush()
    
    def close(self):
        if self.enabled:
            self.muscle_file.close()
            self.eeg_file.close()
            self.classification_file.close()

class SerialThread(threading.Thread):
    """Thread for handling ESP32 serial communication"""
    
    def __init__(self, port=None, debug=False, prefer_bluetooth=False):
        super().__init__(daemon=True)
        self.port = port
        self.debug = debug
        self.prefer_bluetooth = prefer_bluetooth
        self.stop_event = threading.Event()
        
        # Data queues for other threads
        self.muscle_queue = queue.Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.eeg_queue = queue.Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.status_queue = queue.Queue(maxsize=100)
        
        # Serial manager
        self.serial_manager = SerialManager(port=port, debug=debug, prefer_bluetooth=prefer_bluetooth)
        
        # Logger
        self.logger = logging.getLogger('SerialThread')
        
    def run(self):
        """Main thread execution"""
        self.logger.info("Starting serial communication thread")
        
        # Connect to ESP32
        if not self.serial_manager.connect():
            self.logger.error("Failed to connect to ESP32")
            return
        
        # Send initial commands
        self.serial_manager.send_command('START_EEG')
        
        while not self.stop_event.is_set():
            try:
                # Get muscle data
                muscle_data = self.serial_manager.get_muscle_data(timeout=0.01)
                if muscle_data:
                    try:
                        self.muscle_queue.put_nowait(muscle_data)
                    except queue.Full:
                        self.logger.warning("Muscle queue full, dropping data")
                
                # Get EEG data
                eeg_data = self.serial_manager.get_eeg_data(timeout=0.01)
                if eeg_data:
                    try:
                        self.eeg_queue.put_nowait(eeg_data)
                    except queue.Full:
                        self.logger.warning("EEG queue full, dropping data")
                
                # Get status data
                status_data = self.serial_manager.get_status_data(timeout=0.01)
                if status_data:
                    try:
                        self.status_queue.put_nowait(status_data)
                    except queue.Full:
                        pass  # Status queue overflow is not critical
                
                time.sleep(0.001)  # Small delay
                
            except Exception as e:
                self.logger.error(f"Error in serial thread: {e}")
                time.sleep(0.1)
        
        # Cleanup
        self.serial_manager.disconnect()
        self.logger.info("Serial thread stopped")
    
    def stop(self):
        """Stop the thread"""
        self.stop_event.set()

class ProcessingThread(threading.Thread):
    """Thread for EEG processing and classification"""
    
    def __init__(self, eeg_queue, muscle_queue, status_queue):
        super().__init__(daemon=True)
        self.eeg_queue = eeg_queue
        self.muscle_queue = muscle_queue
        self.status_queue = status_queue
        self.stop_event = threading.Event()
        
        # EEG data buffer for windowing
        self.eeg_buffer = deque(maxlen=Config.EEG_WINDOW_SIZE * 2)  # Double buffer size
        
        # Classification results queue
        self.results_queue = queue.Queue(maxsize=100)
        
        # Initialize EEG classifier
        self.classifier = EEGClassifier()
        self.classifier_loaded = False
        
        # Performance monitoring
        self.performance = PerformanceMonitor()
        
        # Data logger
        self.data_logger = DataLogger()
        
        # Logger
        self.logger = logging.getLogger('ProcessingThread')
        
    def run(self):
        """Main thread execution"""
        self.logger.info("Starting processing thread")
        
        # Load trained model
        if self.classifier.load_trained_model(Config.MODEL_PATH):
            self.classifier_loaded = True
            self.logger.info("EEG classifier loaded successfully")
        else:
            self.logger.error("Failed to load EEG classifier")
            return
        
        last_performance_log = time.time()
        
        while not self.stop_event.is_set():
            try:
                # Process muscle data
                self._process_muscle_data()
                
                # Process EEG data
                self._process_eeg_data()
                
                # Process status updates
                self._process_status_data()
                
                # Log performance metrics
                if time.time() - last_performance_log > Config.PERFORMANCE_LOG_INTERVAL:
                    self._log_performance_metrics()
                    last_performance_log = time.time()
                
                time.sleep(0.001)  # Small delay
                
            except Exception as e:
                self.logger.error(f"Error in processing thread: {e}")
                self.performance.record_error()
                time.sleep(0.1)
        
        # Cleanup
        self.data_logger.close()
        self.logger.info("Processing thread stopped")
    
    def _process_muscle_data(self):
        """Process muscle sensor data"""
        try:
            muscle_data = self.muscle_queue.get_nowait()
            
            # Log data
            self.data_logger.log_muscle_data(muscle_data)
            
            # Record performance
            self.performance.record_muscle_sample(muscle_data.timestamp)
            
            # Add to results for WebSocket broadcasting
            muscle_result = {
                'type': 'muscle',
                'timestamp': time.time(),
                'data': {
                    'binary': muscle_data.binary,
                    'decimal': muscle_data.decimal,
                    'esp32_timestamp': muscle_data.timestamp
                }
            }
            
            try:
                self.results_queue.put_nowait(muscle_result)
            except queue.Full:
                pass  # Drop if queue is full
                
        except queue.Empty:
            pass
    
    def _process_eeg_data(self):
        """Process EEG data and perform classification"""
        try:
            eeg_data = self.eeg_queue.get_nowait()
            
            # Log data
            self.data_logger.log_eeg_data(eeg_data)
            
            # Add to EEG buffer
            self.eeg_buffer.extend(eeg_data.channels)
            
            # Record performance
            self.performance.record_eeg_sample(eeg_data.timestamp)
            
            # Check if we have enough data for classification
            if len(self.eeg_buffer) >= Config.EEG_WINDOW_SIZE and self.classifier_loaded:
                # Extract window of data
                window_data = list(self.eeg_buffer)[-Config.EEG_WINDOW_SIZE:]
                
                # Reshape for 12 channels (assuming 128 samples / 12 channels ≈ 10.67 samples per channel)
                # For now, we'll take the last 12 values as a simple approach
                # In a real implementation, you'd want proper windowing per channel
                eeg_channels = eeg_data.channels  # Use current 12-channel reading
                
                # Perform classification
                start_time = time.time()
                classification_result = self.classifier.real_time_predict(eeg_channels)
                
                if 'error' not in classification_result:
                    # Log classification
                    self.data_logger.log_classification(classification_result)
                    
                    # Record performance
                    self.performance.record_classification(classification_result['processing_time'])
                    
                    # Create result for WebSocket
                    eeg_result = {
                        'type': 'classification',
                        'timestamp': time.time(),
                        'data': classification_result
                    }
                    
                    try:
                        self.results_queue.put_nowait(eeg_result)
                    except queue.Full:
                        pass  # Drop if queue is full
                
        except queue.Empty:
            pass
    
    def _process_status_data(self):
        """Process system status updates"""
        try:
            status_data = self.status_queue.get_nowait()
            
            # Create status result for WebSocket
            status_result = {
                'type': 'status',
                'timestamp': time.time(),
                'data': {
                    'eeg_recording': status_data.eeg_recording,
                    'muscle_active': status_data.muscle_active,
                    'uptime': status_data.uptime,
                    'esp32_timestamp': status_data.timestamp
                }
            }
            
            try:
                self.results_queue.put_nowait(status_result)
            except queue.Full:
                pass
                
        except queue.Empty:
            pass
    
    def _log_performance_metrics(self):
        """Log performance metrics"""
        metrics = self.performance.get_metrics()
        self.logger.info(f"Performance: EEG={metrics['eeg_rate']:.1f}Hz, "
                        f"Muscle={metrics['muscle_rate']:.1f}Hz, "
                        f"Classifications={metrics['classification_rate']:.1f}/s, "
                        f"Avg Latency={metrics['avg_classification_latency']:.1f}ms")
    
    def stop(self):
        """Stop the thread"""
        self.stop_event.set()
    
    def get_results_queue(self):
        """Get the results queue for WebSocket thread"""
        return self.results_queue

class WebSocketThread(threading.Thread):
    """Thread for WebSocket server and broadcasting"""
    
    def __init__(self, results_queue):
        super().__init__(daemon=True)
        self.results_queue = results_queue
        self.stop_event = threading.Event()
        
        # WebSocket server (placeholder - will be implemented in Task 19)
        self.clients = []
        self.server = None
        
        # Logger
        self.logger = logging.getLogger('WebSocketThread')
    
    def run(self):
        """Main thread execution"""
        self.logger.info("Starting WebSocket thread")
        self.logger.info("WebSocket server will be implemented in Task 19")
        
        while not self.stop_event.is_set():
            try:
                # Get results to broadcast
                result = self.results_queue.get(timeout=0.1)
                
                # For now, just log the results (WebSocket implementation in Task 19)
                if result['type'] == 'classification':
                    class_data = result['data']
                    self.logger.debug(f"Classification: {class_data['class']} "
                                    f"(confidence: {class_data['confidence']:.2f})")
                elif result['type'] == 'muscle':
                    muscle_data = result['data']
                    if muscle_data['decimal'] > 0:  # Only log when muscles are active
                        self.logger.debug(f"Muscle: {muscle_data['binary']} ({muscle_data['decimal']})")
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in WebSocket thread: {e}")
                time.sleep(0.1)
        
        self.logger.info("WebSocket thread stopped")
    
    def stop(self):
        """Stop the thread"""
        self.stop_event.set()

class MainRealtimeSystem:
    """Main coordinator for the real-time system"""
    
    def __init__(self, args):
        self.args = args
        
        # Configure logging
        log_level = logging.DEBUG if args.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/realtime_system.log') if args.log_data else logging.NullHandler()
            ]
        )
        
        self.logger = logging.getLogger('MainSystem')
        
        # Initialize threads
        self.serial_thread = None
        self.processing_thread = None
        self.websocket_thread = None
        
        # System state
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    def start(self):
        """Start the real-time system"""
        self.logger.info("=== Starting Real-Time EEG Classification System ===")
        
        try:
            # Create data directory
            import os
            os.makedirs(Config.DATA_LOG_DIR, exist_ok=True)
            
            # Initialize and start serial thread
            self.logger.info("Starting serial communication...")
            self.serial_thread = SerialThread(
                port=self.args.port, 
                debug=self.args.debug,
                prefer_bluetooth=getattr(self.args, 'bluetooth', False)
            )
            self.serial_thread.start()
            
            # Wait a moment for serial connection
            time.sleep(2)
            
            # Initialize and start processing thread
            self.logger.info("Starting data processing...")
            self.processing_thread = ProcessingThread(
                self.serial_thread.eeg_queue,
                self.serial_thread.muscle_queue,
                self.serial_thread.status_queue
            )
            self.processing_thread.start()
            
            # Initialize and start WebSocket thread
            self.logger.info("Starting WebSocket server...")
            self.websocket_thread = WebSocketThread(
                self.processing_thread.get_results_queue()
            )
            self.websocket_thread.start()
            
            self.running = True
            self.logger.info("✅ System started successfully!")
            self.logger.info(f"WebSocket server will run on {Config.WEBSOCKET_HOST}:{Config.WEBSOCKET_PORT}")
            self.logger.info("Press Ctrl+C to stop the system")
            
            # Main loop
            self._main_loop()
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            self.stop()
    
    def _main_loop(self):
        """Main system loop"""
        last_status_time = time.time()
        
        while self.running:
            try:
                # Print system status periodically
                if time.time() - last_status_time > 30:  # Every 30 seconds
                    self._print_system_status()
                    last_status_time = time.time()
                
                time.sleep(1)
                
            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(1)
    
    def _print_system_status(self):
        """Print system status summary"""
        if self.processing_thread and hasattr(self.processing_thread, 'performance'):
            metrics = self.processing_thread.performance.get_metrics()
            self.logger.info(f"System Status - Uptime: {metrics['uptime']:.1f}s, "
                           f"EEG: {metrics['eeg_samples_processed']}, "
                           f"Muscle: {metrics['muscle_samples_processed']}, "
                           f"Classifications: {metrics['classifications_made']}")
    
    def stop(self):
        """Stop the real-time system"""
        if not self.running:
            return
        
        self.logger.info("Stopping real-time system...")
        self.running = False
        
        # Stop all threads
        if self.websocket_thread:
            self.websocket_thread.stop()
        
        if self.processing_thread:
            self.processing_thread.stop()
        
        if self.serial_thread:
            self.serial_thread.stop()
        
        # Wait for threads to finish
        threads = [self.serial_thread, self.processing_thread, self.websocket_thread]
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=2)
        
        self.logger.info("✅ System stopped successfully")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Real-Time EEG Classification System')
    parser.add_argument('--port', help='Serial port (auto-detect if not specified)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--log-data', action='store_true', help='Save all data to files')
    parser.add_argument('--demo', action='store_true', help='Use simulated data instead of hardware')
    parser.add_argument('--bluetooth', action='store_true', help='Prefer Bluetooth connection over USB')
    parser.add_argument('--model-path', default=Config.MODEL_PATH, help='Path to trained model')
    parser.add_argument('--websocket-port', type=int, default=Config.WEBSOCKET_PORT, 
                       help='WebSocket server port')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    Config.MODEL_PATH = args.model_path
    Config.WEBSOCKET_PORT = args.websocket_port
    
    if args.demo:
        print("Demo mode not yet implemented why would you want to do that? This is a demo")
        return
    
    # Create and start the system
    system = MainRealtimeSystem(args)
    
    try:
        system.start()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"System error: {e}")
    finally:
        system.stop()

if __name__ == '__main__':
    main() 