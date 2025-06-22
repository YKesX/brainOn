#!/usr/bin/env python3
"""
System Startup Script for Real-Time EEG Classification System
Implements Task 22 from tasks_final.yml

Features:
- Coordinated startup sequence
- Configuration loading and validation
- Health checks for all components
- Error handling and recovery
- Graceful shutdown procedures
- Service management
"""

import os
import sys
import time
import yaml
import logging
import argparse
import threading
import subprocess
import signal
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class SystemStartup:
    """
    Manages the complete system startup and coordination
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.services: Dict[str, Any] = {}
        self.running = False
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger('SystemStartup')
        
        # Service definitions
        self.service_definitions = {
            'websocket_server': {
                'module': 'websocket_server',
                'class': 'BiometricWebSocketServer',
                'priority': 1,
                'critical': False,
                'health_check': self.check_websocket_health
            },
            'main_realtime': {
                'script': 'main_realtime.py',
                'priority': 2,
                'critical': True,
                'health_check': self.check_realtime_health
            }
        }
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/startup.log'),
                logging.StreamHandler()
            ]
        )
    
    def load_configuration(self) -> bool:
        """
        Load and validate system configuration
        Returns True if successful, False otherwise
        """
        try:
            self.logger.info(f"Loading configuration from {self.config_path}")
            
            if not os.path.exists(self.config_path):
                self.logger.error(f"Configuration file not found: {self.config_path}")
                return False
            
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Validate required sections
            required_sections = ['system', 'hardware', 'machine_learning', 'websocket']
            missing_sections = [section for section in required_sections 
                              if section not in self.config]
            
            if missing_sections:
                self.logger.error(f"Missing required configuration sections: {missing_sections}")
                return False
            
            self.logger.info("✅ Configuration loaded and validated")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return False
    
    def validate_environment(self) -> bool:
        """
        Validate the system environment and dependencies
        Returns True if environment is ready, False otherwise
        """
        self.logger.info("Validating system environment...")
        
        checks = [
            ("Python version", self.check_python_version),
            ("Required packages", self.check_python_packages),
            ("Model files", self.check_model_files),
            ("Hardware availability", self.check_hardware),
            ("Directory structure", self.check_directories),
            ("Permissions", self.check_permissions)
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            try:
                if check_func():
                    self.logger.info(f"✅ {check_name}: OK")
                else:
                    self.logger.error(f"❌ {check_name}: FAILED")
                    all_passed = False
            except Exception as e:
                self.logger.error(f"❌ {check_name}: ERROR - {e}")
                all_passed = False
        
        if all_passed:
            self.logger.info("✅ Environment validation passed")
        else:
            self.logger.error("❌ Environment validation failed")
        
        return all_passed
    
    def check_python_version(self) -> bool:
        """Check Python version compatibility"""
        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            return True
        else:
            self.logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
            return False
    
    def check_python_packages(self) -> bool:
        """Check required Python packages"""
        required_packages = [
            'numpy', 'scipy', 'sklearn', 'pandas', 
            'flask', 'flask-socketio', 'flask-cors',
            'pyserial', 'pyyaml'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.logger.error(f"Missing packages: {missing_packages}")
            self.logger.info("Install with: pip install " + " ".join(missing_packages))
            return False
        
        return True
    
    def check_model_files(self) -> bool:
        """Check if trained model files exist"""
        model_path = self.config.get('machine_learning', {}).get('model', {}).get('path')
        
        if not model_path:
            self.logger.warning("No model path specified in configuration")
            return True  # Not critical for startup
        
        if not os.path.exists(model_path):
            self.logger.warning(f"Model file not found: {model_path}")
            self.logger.info("Run eeg_classifier.py to train the model")
            return True  # Not critical for startup
        
        return True
    
    def check_hardware(self) -> bool:
        """Check hardware availability"""
        # Check if we're in demo mode
        if self.config.get('development', {}).get('demo_mode', False):
            self.logger.info("Demo mode enabled - skipping hardware checks")
            return True
        
        # Check for available serial ports
        try:
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            
            if not ports:
                self.logger.warning("No serial ports found")
                return True  # Not critical - might be auto-detected later
            
            self.logger.info(f"Found {len(ports)} serial ports")
            return True
            
        except ImportError:
            self.logger.error("pyserial not available")
            return False
    
    def check_directories(self) -> bool:
        """Check and create required directories"""
        required_dirs = [
            'logs', 'models', 'evaluation', 'examples', 
            'host', 'test', 'src'
        ]
        
        for directory in required_dirs:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory)
                    self.logger.info(f"Created directory: {directory}")
                except Exception as e:
                    self.logger.error(f"Failed to create directory {directory}: {e}")
                    return False
        
        return True
    
    def check_permissions(self) -> bool:
        """Check file and directory permissions"""
        # Check write permissions for log directory
        if not os.access('logs', os.W_OK):
            self.logger.error("No write permission for logs directory")
            return False
        
        # Check if we can create temporary files
        try:
            test_file = 'logs/.permission_test'
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return True
        except Exception as e:
            self.logger.error(f"Permission check failed: {e}")
            return False
    
    def start_services(self) -> bool:
        """
        Start all system services in proper order
        Returns True if all critical services started successfully
        """
        self.logger.info("Starting system services...")
        
        # Sort services by priority
        sorted_services = sorted(
            self.service_definitions.items(),
            key=lambda x: x[1]['priority']
        )
        
        critical_failures = []
        
        for service_name, service_config in sorted_services:
            self.logger.info(f"Starting service: {service_name}")
            
            try:
                if self.start_service(service_name, service_config):
                    self.logger.info(f"✅ {service_name} started successfully")
                else:
                    self.logger.error(f"❌ {service_name} failed to start")
                    if service_config.get('critical', False):
                        critical_failures.append(service_name)
                
                # Wait between service starts
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Exception starting {service_name}: {e}")
                if service_config.get('critical', False):
                    critical_failures.append(service_name)
        
        if critical_failures:
            self.logger.error(f"Critical services failed to start: {critical_failures}")
            return False
        
        self.logger.info("✅ All services started successfully")
        return True
    
    def start_service(self, service_name: str, service_config: Dict[str, Any]) -> bool:
        """Start a single service"""
        try:
            if 'script' in service_config:
                # Start as subprocess
                script_path = service_config['script']
                
                # Build command
                cmd = [sys.executable, script_path]
                
                # Add arguments based on configuration
                if service_name == 'main_realtime':
                    if self.config.get('development', {}).get('debug_mode', False):
                        cmd.append('--debug')
                    if self.config.get('logging', {}).get('data_recording', {}).get('enabled', False):
                        cmd.append('--log-data')
                
                # Start process
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=project_root
                )
                
                # Store service info
                self.services[service_name] = {
                    'type': 'subprocess',
                    'process': process,
                    'config': service_config,
                    'start_time': time.time()
                }
                
                # Check if process started successfully
                time.sleep(1)
                if process.poll() is None:
                    return True
                else:
                    stdout, stderr = process.communicate()
                    self.logger.error(f"Process {service_name} exited immediately:")
                    self.logger.error(f"STDOUT: {stdout.decode()}")
                    self.logger.error(f"STDERR: {stderr.decode()}")
                    return False
            
            elif 'module' in service_config and 'class' in service_config:
                # Start as thread
                module_name = service_config['module']
                class_name = service_config['class']
                
                # Import and instantiate
                module = __import__(module_name)
                service_class = getattr(module, class_name)
                
                # Create instance with configuration
                websocket_config = self.config.get('websocket', {}).get('server', {})
                instance = service_class(
                    host=websocket_config.get('host', 'localhost'),
                    port=websocket_config.get('port', 5000),
                    debug=websocket_config.get('debug', False)
                )
                
                # Start in thread
                thread = threading.Thread(target=instance.run, daemon=True)
                thread.start()
                
                # Store service info
                self.services[service_name] = {
                    'type': 'thread',
                    'thread': thread,
                    'instance': instance,
                    'config': service_config,
                    'start_time': time.time()
                }
                
                return True
            
            else:
                self.logger.error(f"Invalid service configuration for {service_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start service {service_name}: {e}")
            return False
    
    def perform_health_checks(self) -> bool:
        """
        Perform health checks on all running services
        Returns True if all critical services are healthy
        """
        self.logger.info("Performing health checks...")
        
        critical_failures = []
        
        for service_name, service_info in self.services.items():
            service_config = service_info['config']
            
            try:
                health_check_func = service_config.get('health_check')
                if health_check_func:
                    if health_check_func(service_name, service_info):
                        self.logger.info(f"✅ {service_name}: Healthy")
                    else:
                        self.logger.warning(f"⚠️  {service_name}: Unhealthy")
                        if service_config.get('critical', False):
                            critical_failures.append(service_name)
                else:
                    # Basic health check
                    if self.basic_health_check(service_name, service_info):
                        self.logger.info(f"✅ {service_name}: Running")
                    else:
                        self.logger.error(f"❌ {service_name}: Not running")
                        if service_config.get('critical', False):
                            critical_failures.append(service_name)
                            
            except Exception as e:
                self.logger.error(f"Health check failed for {service_name}: {e}")
                if service_config.get('critical', False):
                    critical_failures.append(service_name)
        
        if critical_failures:
            self.logger.error(f"Critical services are unhealthy: {critical_failures}")
            return False
        
        self.logger.info("✅ All health checks passed")
        return True
    
    def basic_health_check(self, service_name: str, service_info: Dict[str, Any]) -> bool:
        """Basic health check for a service"""
        if service_info['type'] == 'subprocess':
            process = service_info['process']
            return process.poll() is None
        elif service_info['type'] == 'thread':
            thread = service_info['thread']
            return thread.is_alive()
        return False
    
    def check_websocket_health(self, service_name: str, service_info: Dict[str, Any]) -> bool:
        """Health check for WebSocket server"""
        try:
            import requests
            websocket_config = self.config.get('websocket', {}).get('server', {})
            host = websocket_config.get('host', 'localhost')
            port = websocket_config.get('port', 5000)
            
            response = requests.get(f'http://{host}:{port}/status', timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def check_realtime_health(self, service_name: str, service_info: Dict[str, Any]) -> bool:
        """Health check for real-time processor"""
        # For now, just check if process is running
        return self.basic_health_check(service_name, service_info)
    
    def start_system(self) -> bool:
        """
        Start the complete system
        Returns True if successful, False otherwise
        """
        self.logger.info("="*60)
        self.logger.info("🚀 STARTING BIOMETRIC AUTHENTICATION SYSTEM")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        # Step 1: Load configuration
        if not self.load_configuration():
            self.logger.error("❌ Configuration loading failed")
            return False
        
        # Step 2: Validate environment
        if not self.validate_environment():
            self.logger.error("❌ Environment validation failed")
            return False
        
        # Step 3: Start services
        if not self.start_services():
            self.logger.error("❌ Service startup failed")
            return False
        
        # Step 4: Health checks
        time.sleep(5)  # Allow services to initialize
        if not self.perform_health_checks():
            self.logger.error("❌ Health checks failed")
            return False
        
        # Step 5: System ready
        startup_time = time.time() - start_time
        self.running = True
        
        self.logger.info("="*60)
        self.logger.info("✅ SYSTEM STARTUP COMPLETED")
        self.logger.info(f"🕐 Startup time: {startup_time:.1f} seconds")
        self.logger.info(f"🌐 WebSocket server: http://localhost:{self.config.get('websocket', {}).get('server', {}).get('port', 5000)}")
        self.logger.info(f"📊 Test client: test/websocket_test.html")
        self.logger.info("="*60)
        
        return True
    
    def stop_system(self):
        """Stop all services gracefully"""
        self.logger.info("🛑 Stopping system services...")
        
        self.running = False
        
        for service_name, service_info in self.services.items():
            try:
                if service_info['type'] == 'subprocess':
                    process = service_info['process']
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.logger.warning(f"Force killing {service_name}")
                        process.kill()
                
                elif service_info['type'] == 'thread':
                    # Threads should be daemon threads and will stop automatically
                    pass
                
                self.logger.info(f"✅ Stopped {service_name}")
                
            except Exception as e:
                self.logger.error(f"Error stopping {service_name}: {e}")
        
        self.logger.info("✅ System shutdown completed")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop_system()
        sys.exit(0)
    
    def run_monitoring_loop(self):
        """Run system monitoring loop"""
        self.logger.info("🔍 Starting system monitoring...")
        
        monitor_interval = self.config.get('monitoring', {}).get('health_checks', {}).get('interval_seconds', 30)
        
        while self.running:
            try:
                time.sleep(monitor_interval)
                
                if not self.perform_health_checks():
                    self.logger.warning("Health checks failed - system may be degraded")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Biometric Authentication System Startup')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--no-monitor', action='store_true', help='Disable monitoring loop')
    parser.add_argument('--health-check-only', action='store_true', help='Perform health checks only')
    
    args = parser.parse_args()
    
    # Create startup manager
    startup = SystemStartup(args.config)
    
    try:
        if args.health_check_only:
            # Just perform health checks
            startup.load_configuration()
            if startup.perform_health_checks():
                print("✅ All health checks passed")
                sys.exit(0)
            else:
                print("❌ Health checks failed")
                sys.exit(1)
        
        # Start the system
        if startup.start_system():
            if not args.no_monitor:
                # Run monitoring loop
                startup.run_monitoring_loop()
            else:
                # Just wait for signals
                while startup.running:
                    time.sleep(1)
        else:
            print("❌ System startup failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        startup.stop_system()

if __name__ == '__main__':
    main() 