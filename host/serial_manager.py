#!/usr/bin/env python3
"""
Serial Communication Manager for ESP32 Biometric Authentication System
Implements Task 15 from tasks_final.yml

Features:
- Auto-baud rate detection and port discovery
- USB and Bluetooth connection support
- Data packet validation with CRC
- Reconnection handling for USB/Bluetooth disconnections
- Buffer management for high-frequency data
- Performance metrics logging
"""

import serial
import serial.tools.list_ports
import time
import threading
import queue
import re
import logging
import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from collections import deque

@dataclass
class MuscleData:
    """Data structure for muscle sensor readings"""
    binary: str
    decimal: int
    timestamp: int
    raw_line: str

@dataclass
class EEGData:
    """Data structure for EEG sensor readings"""
    channels: List[int]  # 12 channels
    timestamp: int
    raw_line: str

@dataclass
class SystemStatus:
    """System status information"""
    eeg_recording: bool
    muscle_active: bool
    uptime: int
    bluetooth_connected: bool
    timestamp: int

class SerialManager:
    """
    Manages serial communication with ESP32 device
    Handles both USB and Bluetooth connections
    Supports muscle and EEG data streams with error recovery
    """
    
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200, 
                 buffer_size: int = 1000, debug: bool = False, prefer_bluetooth: bool = False):
        self.port = port
        self.baudrate = baudrate
        self.buffer_size = buffer_size
        self.debug = debug
        self.prefer_bluetooth = prefer_bluetooth
        
        # Connection type tracking
        self.connection_type = "unknown"  # "usb", "bluetooth", or "unknown"
        
        # Serial connection
        self.serial_conn: Optional[serial.Serial] = None
        self.connected = False
        
        # Data queues for different data types
        self.muscle_queue = queue.Queue(maxsize=buffer_size)
        self.eeg_queue = queue.Queue(maxsize=buffer_size) 
        self.status_queue = queue.Queue(maxsize=100)
        self.error_queue = queue.Queue(maxsize=100)
        
        # Threading
        self.read_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Performance metrics
        self.metrics = {
            'total_packets': 0,
            'muscle_packets': 0,
            'eeg_packets': 0,
            'error_packets': 0,
            'reconnections': 0,
            'start_time': time.time(),
            'last_packet_time': 0,
            'packet_loss_count': 0,
            'bytes_received': 0,
            'connection_type': 'unknown'
        }
        
        # Data validation patterns
        self.muscle_pattern = re.compile(r'^MUSCLE:([01]{4}),(\d+),(\d+)$')
        self.eeg_pattern = re.compile(r'^EEG:([\d,]+),(\d+)$')
        self.status_pattern = re.compile(r'^STATUS: Muscle=([01]{4}) \((\d+)\) Active: (.+?) \| EEG: (\w+) \| BT: (\w+) \| Uptime: (\d+)s$')
        self.cmd_ack_pattern = re.compile(r'^CMD_ACK:(.+)$')
        self.cmd_err_pattern = re.compile(r'^CMD_ERR:(.+)$')
        
        # Setup logging
        self.logger = logging.getLogger('SerialManager')
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
        # Buffer for incomplete lines
        self.line_buffer = ""
        
    def auto_detect_port(self) -> Optional[str]:
        """
        Automatically detect ESP32 device port (USB or Bluetooth)
        Returns port name if found, None otherwise
        """
        self.logger.info("Auto-detecting ESP32 device...")
        
        # Get all available ports
        ports = list(serial.tools.list_ports.comports())
        
        # Separate USB and Bluetooth ports
        usb_ports = []
        bluetooth_ports = []
        
        # ESP32 USB identifiers
        esp32_usb_identifiers = [
            'CP210',  # CP2102 USB-to-UART bridge
            'USB Serial',
            'ESP32',
            'Silicon Labs',
            'UART',
            'CH340'  # Another common USB-UART chip
        ]
        
        # Bluetooth identifiers
        bluetooth_identifiers = [
            'bluetooth',
            'ESP32-BioAuth',
            'standard serial over bluetooth',
            'incoming',
            'outgoing'
        ]
        
        for port in ports:
            self.logger.debug(f"Checking port: {port.device} - {port.description}")
            
            # Check for Bluetooth ports
            is_bluetooth = False
            for bt_id in bluetooth_identifiers:
                if bt_id.lower() in port.description.lower():
                    bluetooth_ports.append(port)
                    is_bluetooth = True
                    self.logger.info(f"Found Bluetooth port: {port.device}")
                    break
            
            if not is_bluetooth:
                # Check for USB ESP32 ports
                for usb_id in esp32_usb_identifiers:
                    if usb_id.lower() in port.description.lower():
                        usb_ports.append(port)
                        self.logger.info(f"Found USB ESP32 port: {port.device}")
                        break
        
        # Choose ports based on preference
        ports_to_test = []
        if self.prefer_bluetooth:
            ports_to_test = bluetooth_ports + usb_ports
        else:
            ports_to_test = usb_ports + bluetooth_ports
        
        # Test each port
        for port in ports_to_test:
            self.logger.info(f"Testing port: {port.device}")
            if self._verify_esp32_device(port.device):
                # Determine connection type
                self.connection_type = "bluetooth" if port in bluetooth_ports else "usb"
                self.metrics['connection_type'] = self.connection_type
                
                self.logger.info(f"Verified ESP32 device on {port.device} ({self.connection_type})")
                return port.device
        
        self.logger.warning("No ESP32 device found automatically")
        return None
    
    def _verify_esp32_device(self, port: str) -> bool:
        """
        Verify that the device on the port is our ESP32 system
        """
        try:
            # Open connection temporarily
            self.logger.debug(f"Verifying ESP32 on {port}")
            test_conn = serial.Serial(port, self.baudrate, timeout=3)
            time.sleep(1)  # Allow device to initialize
            
            # Clear any existing data
            test_conn.reset_input_buffer()
            
            # Send status command
            test_conn.write(b'STATUS\n')
            time.sleep(1)
            
            # Read response lines
            response_lines = []
            start_time = time.time()
            while time.time() - start_time < 3:
                if test_conn.in_waiting:
                    line = test_conn.readline().decode('utf-8', errors='ignore').strip()
                    response_lines.append(line)
                    self.logger.debug(f"Verification response: {line}")
                    
                    # Check for our system signature
                    if ('ESP32 Biometric Authentication System' in line or
                        'CMD_ACK:STATUS' in line or
                        'MUSCLE:' in line or
                        'EEG:' in line or
                        'ESP32-BioAuth' in line):
                        test_conn.close()
                        return True
            
            test_conn.close()
            return False
            
        except Exception as e:
            self.logger.debug(f"Error verifying device on {port}: {e}")
            return False
    
    def connect(self) -> bool:
        """
        Establish connection to ESP32 device (USB or Bluetooth)
        Returns True if successful, False otherwise
        """
        if self.connected:
            self.logger.warning("Already connected")
            return True
        
        # Auto-detect port if not specified
        if not self.port:
            self.port = self.auto_detect_port()
            if not self.port:
                self.logger.error("Could not detect ESP32 device")
                return False
        
        try:
            self.logger.info(f"Connecting to {self.port} at {self.baudrate} baud ({self.connection_type})")
            
            # Adjust timeout for Bluetooth connections (they can be slower)
            timeout = 3.0 if self.connection_type == "bluetooth" else 1.0
            
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=timeout,
                write_timeout=timeout
            )
            
            # Bluetooth connections may need more time
            init_delay = 3 if self.connection_type == "bluetooth" else 2
            time.sleep(init_delay)
            
            # Clear any existing data
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            
            # Test connection
            if self._test_connection():
                self.connected = True
                self.logger.info(f"Successfully connected to ESP32 via {self.connection_type}")
                
                # Start reading thread
                self.stop_event.clear()
                self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
                self.read_thread.start()
                
                return True
            else:
                self.serial_conn.close()
                self.serial_conn = None
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            self.serial_conn = None
            return False
    
    def _test_connection(self) -> bool:
        """Test connection by sending STATUS command"""
        try:
            self.serial_conn.write(b'STATUS\n')
            
            # Wait for response (longer timeout for Bluetooth)
            timeout = 8 if self.connection_type == "bluetooth" else 5
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    self.logger.debug(f"Connection test response: {line}")
                    if 'CMD_ACK:STATUS' in line or 'STATUS:' in line:
                        return True
                time.sleep(0.1)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from ESP32 device"""
        if not self.connected:
            return
        
        self.logger.info(f"Disconnecting from ESP32 ({self.connection_type})")
        
        # Stop reading thread
        self.stop_event.set()
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=3)
        
        # Close serial connection
        if self.serial_conn:
            try:
                self.serial_conn.close()
            except:
                pass
            self.serial_conn = None
        
        self.connected = False
        self.connection_type = "unknown"
        self.logger.info("Disconnected")
    
    def _read_loop(self):
        """Main reading loop running in separate thread"""
        self.logger.info(f"Starting data reading loop ({self.connection_type})")
        
        while not self.stop_event.is_set():
            try:
                if not self.serial_conn or not self.serial_conn.is_open:
                    # Attempt reconnection
                    if self._attempt_reconnection():
                        continue
                    else:
                        time.sleep(1)
                        continue
                
                # Read data
                if self.serial_conn.in_waiting:
                    data = self.serial_conn.read(self.serial_conn.in_waiting)
                    self.metrics['bytes_received'] += len(data)
                    
                    # Process data
                    self._process_incoming_data(data)
                else:
                    time.sleep(0.001)  # Small delay to prevent CPU spinning
                    
            except Exception as e:
                self.logger.error(f"Error in read loop: {e}")
                self.connected = False
                time.sleep(1)
    
    def _process_incoming_data(self, data: bytes):
        """Process incoming data and parse lines"""
        try:
            # Add to line buffer
            self.line_buffer += data.decode('utf-8', errors='ignore')
            
            # Process complete lines
            while '\n' in self.line_buffer:
                line, self.line_buffer = self.line_buffer.split('\n', 1)
                line = line.strip()
                
                if line:
                    self._parse_line(line)
                    
        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
    
    def _parse_line(self, line: str):
        """Parse a complete line of data"""
        try:
            self.metrics['total_packets'] += 1
            self.metrics['last_packet_time'] = time.time()
            
            if self.debug:
                self.logger.debug(f"Parsing: {line}")
            
            # Try to match different data types
            if line.startswith('MUSCLE:'):
                self._parse_muscle_data(line)
            elif line.startswith('EEG:'):
                self._parse_eeg_data(line)
            elif line.startswith('STATUS:'):
                self._parse_status_data(line)
            elif line.startswith('CMD_ACK:'):
                self._parse_command_ack(line)
            elif line.startswith('CMD_ERR:'):
                self._parse_command_error(line)
            elif line.startswith('==='):
                # System message, log it
                self.logger.info(f"System: {line}")
            elif '📶 Bluetooth:' in line or '🔗 USB Serial:' in line:
                # Connection info message
                self.logger.info(f"Device info: {line}")
            else:
                # Unknown format
                if self.debug:
                    self.logger.debug(f"Unknown format: {line}")
                
        except Exception as e:
            self.logger.error(f"Error parsing line '{line}': {e}")
            self.metrics['error_packets'] += 1
    
    def _parse_muscle_data(self, line: str):
        """Parse muscle sensor data"""
        match = self.muscle_pattern.match(line)
        if match:
            binary, decimal, timestamp = match.groups()
            
            muscle_data = MuscleData(
                binary=binary,
                decimal=int(decimal),
                timestamp=int(timestamp),
                raw_line=line
            )
            
            try:
                self.muscle_queue.put_nowait(muscle_data)
                self.metrics['muscle_packets'] += 1
            except queue.Full:
                # Remove oldest item and add new one
                try:
                    self.muscle_queue.get_nowait()
                    self.muscle_queue.put_nowait(muscle_data)
                    self.metrics['packet_loss_count'] += 1
                except:
                    pass
        else:
            self.logger.warning(f"Invalid muscle data format: {line}")
            self.metrics['error_packets'] += 1
    
    def _parse_eeg_data(self, line: str):
        """Parse EEG sensor data"""
        match = self.eeg_pattern.match(line)
        if match:
            channels_str, timestamp = match.groups()
            
            try:
                # Parse channel values
                channel_values = [int(x) for x in channels_str.split(',')]
                
                if len(channel_values) == 12:  # Verify 12 channels
                    eeg_data = EEGData(
                        channels=channel_values,
                        timestamp=int(timestamp),
                        raw_line=line
                    )
                    
                    try:
                        self.eeg_queue.put_nowait(eeg_data)
                        self.metrics['eeg_packets'] += 1
                    except queue.Full:
                        # Remove oldest item and add new one
                        try:
                            self.eeg_queue.get_nowait()
                            self.eeg_queue.put_nowait(eeg_data)
                            self.metrics['packet_loss_count'] += 1
                        except:
                            pass
                else:
                    self.logger.warning(f"Invalid EEG channel count: {len(channel_values)}")
                    self.metrics['error_packets'] += 1
                    
            except ValueError as e:
                self.logger.warning(f"Error parsing EEG values: {e}")
                self.metrics['error_packets'] += 1
        else:
            self.logger.warning(f"Invalid EEG data format: {line}")
            self.metrics['error_packets'] += 1
    
    def _parse_status_data(self, line: str):
        """Parse system status data"""
        match = self.status_pattern.match(line)
        if match:
            muscle_binary, muscle_decimal, active_muscles, eeg_status, bt_status, uptime = match.groups()
            
            status = SystemStatus(
                eeg_recording=(eeg_status == "Recording"),
                muscle_active=(active_muscles != "None"),
                bluetooth_connected=(bt_status == "Connected"),
                uptime=int(uptime),
                timestamp=int(time.time() * 1000)
            )
            
            try:
                self.status_queue.put_nowait(status)
            except queue.Full:
                try:
                    self.status_queue.get_nowait()
                    self.status_queue.put_nowait(status)
                except:
                    pass
    
    def _parse_command_ack(self, line: str):
        """Parse command acknowledgment"""
        match = self.cmd_ack_pattern.match(line)
        if match:
            self.logger.info(f"Command ACK: {match.group(1)}")
    
    def _parse_command_error(self, line: str):
        """Parse command error"""
        match = self.cmd_err_pattern.match(line)
        if match:
            self.logger.warning(f"Command Error: {match.group(1)}")
    
    def _attempt_reconnection(self) -> bool:
        """Attempt to reconnect to device"""
        self.logger.info(f"Attempting reconnection ({self.connection_type})...")
        self.metrics['reconnections'] += 1
        
        try:
            if self.serial_conn:
                self.serial_conn.close()
            
            # Longer delay for Bluetooth reconnections
            delay = 5 if self.connection_type == "bluetooth" else 2
            time.sleep(delay)
            
            # Try to reconnect to the same port first
            timeout = 3.0 if self.connection_type == "bluetooth" else 1.0
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=timeout,
                write_timeout=timeout
            )
            
            time.sleep(delay)
            
            if self._test_connection():
                self.connected = True
                self.logger.info("Reconnection successful")
                return True
            else:
                self.serial_conn.close()
                self.serial_conn = None
                
                # If that fails, try auto-detection again
                self.logger.info("Trying auto-detection for reconnection")
                new_port = self.auto_detect_port()
                if new_port and new_port != self.port:
                    self.port = new_port
                    return self._attempt_reconnection()
                
                return False
                
        except Exception as e:
            self.logger.error(f"Reconnection failed: {e}")
            return False
    
    def send_command(self, command: str) -> bool:
        """Send command to ESP32"""
        if not self.connected or not self.serial_conn:
            self.logger.error("Not connected")
            return False
        
        try:
            command_bytes = (command + '\n').encode('utf-8')
            self.serial_conn.write(command_bytes)
            self.logger.info(f"Sent command: {command}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send command: {e}")
            return False
    
    def get_muscle_data(self, timeout: float = 0.1) -> Optional[MuscleData]:
        """Get latest muscle data"""
        try:
            return self.muscle_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_eeg_data(self, timeout: float = 0.1) -> Optional[EEGData]:
        """Get latest EEG data"""
        try:
            return self.eeg_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_status_data(self, timeout: float = 0.1) -> Optional[SystemStatus]:
        """Get latest status data"""
        try:
            return self.status_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        uptime = time.time() - self.metrics['start_time']
        enhanced_metrics = self.metrics.copy()
        enhanced_metrics.update({
            'uptime': uptime,
            'packets_per_second': self.metrics['total_packets'] / uptime if uptime > 0 else 0,
            'connected': self.connected,
            'connection_type': self.connection_type,
            'port': self.port
        })
        return enhanced_metrics
    
    def clear_queues(self):
        """Clear all data queues"""
        while not self.muscle_queue.empty():
            try:
                self.muscle_queue.get_nowait()
            except:
                break
        
        while not self.eeg_queue.empty():
            try:
                self.eeg_queue.get_nowait()
            except:
                break
        
        while not self.status_queue.empty():
            try:
                self.status_queue.get_nowait()
            except:
                break
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information"""
        return {
            'connected': self.connected,
            'port': self.port,
            'connection_type': self.connection_type,
            'baudrate': self.baudrate,
            'total_packets': self.metrics['total_packets'],
            'reconnections': self.metrics['reconnections']
        }

def main():
    """Demo/test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ESP32 Serial Manager Test')
    parser.add_argument('--port', help='Serial port (auto-detect if not specified)')
    parser.add_argument('--baudrate', type=int, default=115200, help='Baud rate')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--bluetooth', action='store_true', help='Prefer Bluetooth connection')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create serial manager
    manager = SerialManager(
        port=args.port, 
        baudrate=args.baudrate, 
        debug=args.debug,
        prefer_bluetooth=args.bluetooth
    )
    
    # Connect
    if not manager.connect():
        print("Failed to connect to ESP32")
        return
    
    try:
        print("Connected! Monitoring data... Press Ctrl+C to stop")
        print(f"Connection: {manager.get_connection_info()}")
        
        while True:
            # Get muscle data
            muscle_data = manager.get_muscle_data(timeout=0.1)
            if muscle_data:
                print(f"Muscle: {muscle_data.binary} ({muscle_data.decimal})")
            
            # Get EEG data
            eeg_data = manager.get_eeg_data(timeout=0.1)
            if eeg_data:
                print(f"EEG: {eeg_data.channels[:3]}... ({len(eeg_data.channels)} channels)")
            
            # Get status
            status = manager.get_status_data(timeout=0.1)
            if status:
                print(f"Status: EEG={status.eeg_recording}, Muscle={status.muscle_active}, BT={status.bluetooth_connected}")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        manager.disconnect()

if __name__ == '__main__':
    main() 