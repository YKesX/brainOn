#!/usr/bin/env python3
"""
Real-Time EEG Classification System
Multi-Modal Biometric Authentication: Live EEG Brainwave Classification

"""

import os
import sys
import time
import numpy as np
import pandas as pd
import pickle
import serial
import serial.tools.list_ports
import threading
import queue
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Import trained models
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. CNN functionality will be disabled.")
    TENSORFLOW_AVAILABLE = False

class RealTimeEEGClassifier:
    """
    Real-Time EEG Classification System
    Connects to ESP32, processes live EEG data, and classifies brain states
    Classes: 0=Baseline, 1=Unlock, 2=Transaction
    """
    
    def __init__(self, model_path='models/', port=None, baud_rate=115200, demo_mode=False):
        self.model_path = model_path
        self.port = port
        self.baud_rate = baud_rate
        self.demo_mode = demo_mode
        
        # EEG Configuration
        self.eeg_channels = ['EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 
                            'EEG7', 'EEG8', 'EEG9', 'EEG10', 'EEG11', 'EEG12']
        self.class_names = ['Baseline', 'Unlock', 'Transaction']
        
        # Models and preprocessing
        self.rf_model = None
        self.cnn_model = None
        self.scaler = None
        
        # Serial communication
        self.serial_connection = None
        self.is_connected = False
        self.data_queue = queue.Queue(maxsize=1000)
        
        # Real-time processing
        self.window_size = 50  # Number of samples for feature extraction
        self.eeg_buffer = deque(maxlen=self.window_size)
        self.prediction_history = deque(maxlen=10)  # For temporal smoothing
        
        # Threading control
        self.stop_threads = threading.Event()
        self.reader_thread = None
        self.processor_thread = None
        
        # Demo mode variables
        self.demo_data = None
        self.demo_index = 0
        
        print("=== Task 11: Serial Communication Setup ===")
        if demo_mode:
            print("🎮 DEMO MODE: Using simulated EEG data")
    
    def load_demo_data(self):
        """Load demo data from the CSV file for simulation"""
        try:
            df = pd.read_csv('examples/eeg_patterns.csv')
            self.demo_data = df[self.eeg_channels].values
            print(f"✅ Demo data loaded: {len(self.demo_data)} samples")
            return True
        except Exception as e:
            print(f"❌ Failed to load demo data: {str(e)}")
            return False
    
    def generate_demo_eeg_data(self):
        """Generate simulated EEG data for demo mode"""
        if self.demo_data is None:
            if not self.load_demo_data():
                # Generate random data if CSV loading fails
                return [2000 + np.random.randint(-100, 100) for _ in range(12)]
        
        # Cycle through demo data
        if self.demo_index >= len(self.demo_data):
            self.demo_index = 0
        
        sample = self.demo_data[self.demo_index].tolist()
        self.demo_index += 1
        
        # Add some noise to make it more realistic
        noisy_sample = [int(val + np.random.normal(0, 10)) for val in sample]
        return noisy_sample
    
    def detect_serial_ports(self):
        """
        Requirements:
        - Serial port detection and configuration
        - Error handling for connection issues
        """
        
        if self.demo_mode:
            print("🎮 Demo mode: Skipping serial port detection")
            return "DEMO_PORT"
        
        print("Detecting available serial ports...")
        
        ports = serial.tools.list_ports.comports()
        available_ports = []
        
        for port in ports:
            print(f"  Found: {port.device} - {port.description}")
            available_ports.append(port.device)
        
        if not available_ports:
            print("❌ No serial ports detected")
            return None
        
        # Try to auto-detect ESP32
        esp32_keywords = ['USB', 'Serial', 'ESP32', 'CP210', 'CH340']
        for port in ports:
            for keyword in esp32_keywords:
                if keyword.lower() in port.description.lower():
                    print(f"✓ Potential ESP32 detected: {port.device}")
                    return port.device
        
        # If no auto-detection, return first available port
        print(f"⚠ Auto-detection failed. Using first port: {available_ports[0]}")
        return available_ports[0]
    
    def establish_connection(self, port=None):
        """
        - Serial port detection and configuration
        - Parse incoming EEG data (12 channels + timestamps)
        - Error handling for connection issues
        - Data validation and filtering
        """
        
        if self.demo_mode:
            print("🎮 Demo mode: Simulating ESP32 connection")
            self.is_connected = True
            return True
        
        if port is None:
            port = self.port or self.detect_serial_ports()
        
        if port is None:
            raise ConnectionError("No suitable serial port found")
        
        try:
            print(f"Attempting to connect to {port} at {self.baud_rate} baud...")
            
            self.serial_connection = serial.Serial(
                port=port,
                baudrate=self.baud_rate,
                timeout=1,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            
            # Wait for connection to stabilize
            time.sleep(2)
            
            # Clear any initial garbage data
            self.serial_connection.flushInput()
            
            # Test connection by reading a few lines
            print("Testing connection...")
            for i in range(5):
                try:
                    line = self.serial_connection.readline().decode('utf-8').strip()
                    if line:
                        print(f"  Test read {i+1}: {line[:50]}...")
                except:
                    print(f"  Test read {i+1}: Failed")
            
            self.is_connected = True
            print(f"✅ Successfully connected to {port}")
            return True
            
        except serial.SerialException as e:
            print(f"❌ Failed to connect to {port}: {str(e)}")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            self.is_connected = False
            return False
    
    def parse_eeg_data(self, line):
        """
        Parse incoming EEG data line from ESP32
        Expected format: "Muscle readings: 1842, 1851 | EEG readings: 2034, 1987, 2123, ..."
        """
        try:
            # Look for EEG data in the line
            if "EEG readings:" in line:
                # Extract EEG part
                eeg_part = line.split("EEG readings:")[1].strip()
                
                # Parse comma-separated values
                eeg_values = []
                for val_str in eeg_part.split(','):
                    val_str = val_str.strip()
                    if val_str.isdigit():
                        eeg_values.append(int(val_str))
                
                # Validate we have 12 EEG channels
                if len(eeg_values) == 12:
                    return {
                        'timestamp': time.time(),
                        'eeg_data': eeg_values,
                        'valid': True
                    }
                else:
                    return {'valid': False, 'error': f'Expected 12 channels, got {len(eeg_values)}'}
            
            return {'valid': False, 'error': 'No EEG data found'}
            
        except Exception as e:
            return {'valid': False, 'error': f'Parsing error: {str(e)}'}
    
    def data_reader_thread(self):
        """
        Task 11: Background thread for continuous data reading
        
        Requirements:
        - Buffer management for continuous data stream
        - Data validation and filtering
        """
        
        print("🔄 Starting data reader thread...")
        
        while not self.stop_threads.is_set() and self.is_connected:
            try:
                if self.demo_mode:
                    # Generate demo EEG data
                    eeg_data = self.generate_demo_eeg_data()
                    parsed_data = {
                        'timestamp': time.time(),
                        'eeg_data': eeg_data,
                        'valid': True
                    }
                    
                    # Add to queue for processing
                    try:
                        self.data_queue.put(parsed_data, block=False)
                    except queue.Full:
                        # Remove oldest item if queue is full
                        try:
                            self.data_queue.get(block=False)
                            self.data_queue.put(parsed_data, block=False)
                        except queue.Empty:
                            pass
                    
                    time.sleep(0.1)  # 10Hz demo data rate
                    
                else:
                    # Real serial communication
                    if self.serial_connection and self.serial_connection.in_waiting > 0:
                        line = self.serial_connection.readline().decode('utf-8', errors='ignore').strip()
                        
                        if line:
                            # Parse EEG data
                            parsed_data = self.parse_eeg_data(line)
                            
                            if parsed_data['valid']:
                                # Add to queue for processing
                                try:
                                    self.data_queue.put(parsed_data, block=False)
                                except queue.Full:
                                    # Remove oldest item if queue is full
                                    try:
                                        self.data_queue.get(block=False)
                                        self.data_queue.put(parsed_data, block=False)
                                    except queue.Empty:
                                        pass
                            
                    time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                print(f"❌ Data reader error: {str(e)}")
                break
        
        print("🛑 Data reader thread stopped")
    
    def load_trained_models(self):
        """
        Task 12: Load pre-trained models for real-time classification
        
        Requirements:
        - Load both RF and CNN models
        - Load preprocessing parameters
        """
        
        print("\n=== Task 12: Loading Trained Models ===")
        
        # Load Random Forest model
        rf_path = os.path.join(self.model_path, 'rf_eeg_classifier.pkl')
        if os.path.exists(rf_path):
            try:
                with open(rf_path, 'rb') as f:
                    rf_data = pickle.load(f)
                    self.rf_model = rf_data['model']
                    self.scaler = rf_data['scaler']
                    feature_names = rf_data['feature_names']
                    class_names = rf_data['class_names']
                
                print(f"✅ Random Forest model loaded successfully")
                print(f"  Features: {len(feature_names)} channels")
                print(f"  Classes: {class_names}")
                
            except Exception as e:
                print(f"❌ Failed to load Random Forest model: {str(e)}")
                return False
        else:
            print(f"❌ Random Forest model not found: {rf_path}")
            return False
        
        # Load CNN model (if available)
        cnn_path = os.path.join(self.model_path, 'cnn_eeg_classifier.h5')
        if TENSORFLOW_AVAILABLE and os.path.exists(cnn_path):
            try:
                self.cnn_model = tf.keras.models.load_model(cnn_path)
                print(f"✅ CNN model loaded successfully")
                print(f"  Input shape: {self.cnn_model.input_shape}")
                print(f"  Output shape: {self.cnn_model.output_shape}")
                
            except Exception as e:
                print(f"❌ Failed to load CNN model: {str(e)}")
                self.cnn_model = None
        else:
            print(f"⚠ CNN model not available")
            self.cnn_model = None
        
        print("✅ Task 12 COMPLETED: Models loaded for real-time classification")
        return True
    
    def extract_features_realtime(self, eeg_samples):
        """
        - Apply same preprocessing as training data
        - Real-time feature normalization using training statistics
        - Feature vector formatting for model input
        """
        
        if len(eeg_samples) < self.window_size:
            return None
        
        # Convert to numpy array
        eeg_array = np.array(eeg_samples)
        
        # Take the most recent window_size samples
        recent_samples = eeg_array[-self.window_size:, :]
        
        # Simple feature extraction: mean of each channel over the window
        # This matches our training data format (one sample per channel)
        features = np.mean(recent_samples, axis=0)
        
        # Apply same scaling as training data
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1))
            return features.flatten()
        
        return features
    
    def classify_eeg_state(self, features):
        """
        - Ensemble prediction (weighted average)
        - Confidence scoring for predictions
        - Temporal smoothing to reduce noise
        """
        
        if features is None or self.rf_model is None:
            return None
        
        features = features.reshape(1, -1)
        predictions = {}
        
        # Random Forest prediction
        try:
            rf_pred = self.rf_model.predict(features)[0]
            rf_proba = self.rf_model.predict_proba(features)[0]
            predictions['rf'] = {
                'class': rf_pred,
                'probabilities': rf_proba,
                'confidence': np.max(rf_proba)
            }
        except Exception as e:
            print(f"RF prediction error: {str(e)}")
            predictions['rf'] = None
        
        # CNN prediction (if available)
        if self.cnn_model is not None:
            try:
                cnn_proba = self.cnn_model.predict(features, verbose=0)[0]
                cnn_pred = np.argmax(cnn_proba)
                predictions['cnn'] = {
                    'class': cnn_pred,
                    'probabilities': cnn_proba,
                    'confidence': np.max(cnn_proba)
                }
            except Exception as e:
                print(f"CNN prediction error: {str(e)}")
                predictions['cnn'] = None
        
        # Ensemble prediction (weighted average)
        if predictions['rf'] and predictions['cnn']:
            # Weight: 0.6 for RF, 0.4 for CNN (since RF typically more reliable on small datasets)
            ensemble_proba = 0.6 * predictions['rf']['probabilities'] + 0.4 * predictions['cnn']['probabilities']
            ensemble_pred = np.argmax(ensemble_proba)
            ensemble_conf = np.max(ensemble_proba)
        elif predictions['rf']:
            ensemble_proba = predictions['rf']['probabilities']
            ensemble_pred = predictions['rf']['class']
            ensemble_conf = predictions['rf']['confidence']
        else:
            return None
        
        # Temporal smoothing
        self.prediction_history.append(ensemble_pred)
        
        # Use majority vote for smoothing
        if len(self.prediction_history) >= 3:
            recent_predictions = list(self.prediction_history)[-3:]
            smoothed_pred = max(set(recent_predictions), key=recent_predictions.count)
        else:
            smoothed_pred = ensemble_pred
        
        return {
            'predicted_class': smoothed_pred,
            'class_name': self.class_names[smoothed_pred],
            'confidence': ensemble_conf,
            'probabilities': ensemble_proba,
            'raw_prediction': ensemble_pred,
            'individual_models': predictions
        }
    
    def data_processor_thread(self):
        """
        - Real-time feature extraction
        - Live classification
        - Classification result logging
        """
        
        print("🔄 Starting data processor thread...")
        
        while not self.stop_threads.is_set():
            try:
                # Get data from queue (with timeout)
                try:
                    data = self.data_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Add to EEG buffer
                self.eeg_buffer.append(data['eeg_data'])
                
                # Extract features when we have enough samples
                if len(self.eeg_buffer) >= self.window_size:
                    features = self.extract_features_realtime(list(self.eeg_buffer))
                    
                    if features is not None:
                        # Classify the current state
                        result = self.classify_eeg_state(features)
                        
                        if result:
                            # Print classification result
                            print(f"🧠 EEG State: {result['class_name']} "
                                 f"(Confidence: {result['confidence']:.2f}) "
                                 f"[{time.strftime('%H:%M:%S')}]")
                            
                            # Optional: Save to log file
                            self.log_classification_result(result)
                
            except Exception as e:
                print(f"❌ Data processor error: {str(e)}")
                time.sleep(0.1)
        
        print("🛑 Data processor thread stopped")
    
    def log_classification_result(self, result):
        """Log classification results to file for analysis"""
        log_entry = {
            'timestamp': time.time(),
            'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
            'predicted_class': result['predicted_class'],
            'class_name': result['class_name'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'].tolist()
        }
        
        # Append to log file
        log_file = 'real_time_classifications.log'
        with open(log_file, 'a') as f:
            f.write(f"{log_entry}\n")
    
    def start_real_time_classification(self):
        """
        - <100ms prediction latency
        - Stable results
        """
        
        print("\n=== Task 13: Starting Real-Time Classification ===")
        
        if not self.is_connected:
            print("❌ No serial connection established")
            return False
        
        if self.rf_model is None:
            print("❌ No trained models loaded")
            return False
        
        print("🚀 Starting real-time EEG classification...")
        print("Press Ctrl+C to stop")
        
        # Start background threads
        self.reader_thread = threading.Thread(target=self.data_reader_thread)
        self.processor_thread = threading.Thread(target=self.data_processor_thread)
        
        self.reader_thread.start()
        self.processor_thread.start()
        
        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
                
                # Print status every 30 seconds
                if int(time.time()) % 30 == 0:
                    print(f"📊 System status: Queue size: {self.data_queue.qsize()}, "
                         f"Buffer size: {len(self.eeg_buffer)}")
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping real-time classification...")
            self.stop_real_time_classification()
            return True
    
    def stop_real_time_classification(self):
        """Stop all threads and close connections"""
        
        print("Stopping threads...")
        self.stop_threads.set()
        
        # Wait for threads to finish
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=2)
        
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=2)
        
        # Close serial connection
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            print("✅ Serial connection closed")
        
        print("✅ Real-time classification stopped")

def main():
    """Main execution for real-time EEG classification"""
    print("Real-Time EEG Classification System")
    print("Multi-Modal Biometric Authentication")
    print("="*50)
    
    # Check if demo mode is requested
    demo_mode = '--demo' in sys.argv or '--simulation' in sys.argv
    
    # Initialize classifier
    classifier = RealTimeEEGClassifier(demo_mode=demo_mode)
    
    try:
        # Task 11: Establish Serial Connection
        if not classifier.establish_connection():
            print("❌ Failed to establish serial connection")
            return False
        
        # Task 12: Load Trained Models
        if not classifier.load_trained_models():
            print("❌ Failed to load trained models")
            return False
        
        # Task 13: Start Real-Time Classification
        classifier.start_real_time_classification()
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        classifier.stop_real_time_classification()
    
    return True

if __name__ == "__main__":
    main() 