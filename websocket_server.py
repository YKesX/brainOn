#!/usr/bin/env python3
"""
Flask WebSocket Server for Real-Time EEG Classification Broadcasting

Features:
- Two operational modes: demo, real
- Flask-SocketIO for WebSocket support
- Hash-based classification output for crypto wallet integration
- Client connection management
- CORS support for web applications
- REST API endpoints for status and data access
"""

import os
import sys
import time
import json
import logging
import threading
import queue
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

import flask
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, disconnect
from flask_cors import CORS

# Hash mappings for classification modes
CLASS_HASH_MAPPING = {
    0: "f7b11509f4d675c3c44f0dd37ca830bb02e8cfa58f04c46283c4bfcbdce1ff45",  # Baseline
    1: "8988fb4fc735b6dc5d3b0acad50edf57e5fcf1ff69891940ce2c0ce4490d4ed9",  # Unlock
    2: "a18ac4e6fbd3fc024a07a21dafbac37d828ca8a04a0e34f368f1ec54e0d4fffb"   # Transaction
}

CLASS_NAMES = {
    0: "Baseline",
    1: "Unlock", 
    2: "Transaction"
}

@dataclass
class ClassificationResult:
    """Data structure for EEG classification results"""
    timestamp: float
    sequence: int
    classification: Dict[str, Any]
    muscle_state: Dict[str, Any]
    system: Dict[str, Any]

@dataclass
class ClientInfo:
    """Information about connected WebSocket clients"""
    session_id: str
    remote_addr: str
    user_agent: str
    connected_at: float
    last_ping: float
    authenticated: bool = False
    client_type: str = "unknown"

class BiometricWebSocketServer:
    """
    WebSocket server for real-time biometric authentication data broadcasting
    Supports two operational modes: demo and real
    """
    
    def __init__(self, host='localhost', port=5000, debug=False, mode='demo'):
        self.host = host
        self.port = port
        self.debug = debug
        self.mode = mode  # 'demo' or 'real'
        
        # Flask app setup
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'biometric_auth_secret_key_change_in_production'
        
        # Enable CORS for cross-origin requests (crypto wallet integration)
        CORS(self.app, origins="*")
        
        # SocketIO setup
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            logger=debug,
            engineio_logger=debug,
            async_mode='threading'
        )
        
        # Client management
        self.connected_clients: Dict[str, ClientInfo] = {}
        self.client_lock = threading.Lock()
        
        # Data storage
        self.classification_history: List[ClassificationResult] = []
        self.current_state: Optional[ClassificationResult] = None
        self.max_history_size = 1000
        
        # Performance metrics
        self.metrics = {
            'server_start_time': time.time(),
            'total_connections': 0,
            'current_connections': 0,
            'messages_sent': 0,
            'classification_updates': 0,
            'muscle_updates': 0,
            'errors': 0
        }
        
        # Sequence counter for message ordering
        self.sequence_counter = 0
        self.sequence_lock = threading.Lock()
        
        # Mode-specific variables
        self.eeg_classifier = None  # Will be set for real mode
        
        # Logger
        self.logger = logging.getLogger('WebSocketServer')
        
        # Setup routes and event handlers
        self._setup_routes()
        self._setup_socket_events()
        
        self.logger.info(f"WebSocket server initialized on {host}:{port} in {mode} mode")
    
    def set_mode(self, mode: str):
        """Change operational mode"""
        if mode in ['demo', 'real']:
            self.mode = mode
            self.logger.info(f"Mode changed to: {mode}")
            self.broadcast_system_status({'mode': mode, 'message': f'Server mode changed to {mode}'})
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'demo' or 'real'")
    
    def set_eeg_classifier(self, classifier):
        """Set EEG classifier for real mode"""
        self.eeg_classifier = classifier
        self.logger.info("EEG classifier set for real mode")
    
    def _convert_class_id_to_hash(self, class_id: int) -> str:
        """Convert class ID to corresponding hash string"""
        return CLASS_HASH_MAPPING.get(class_id, CLASS_HASH_MAPPING[0])  # Default to baseline
    
    def _setup_routes(self):
        """Setup REST API endpoints"""
        
        @self.app.route('/')
        def index():
            """Main page with WebSocket connection info"""
            return {
                'service': 'Biometric Authentication WebSocket Server',
                'version': '2.0.0',
                'websocket_url': f'ws://{self.host}:{self.port}',
                'status': 'running',
                'mode': self.mode,
                'clients_connected': len(self.connected_clients),
                'uptime': time.time() - self.metrics['server_start_time']
            }
        
        @self.app.route('/status')
        def status():
            """Server health check endpoint"""
            return {
                'status': 'healthy',
                'timestamp': time.time(),
                'mode': self.mode,
                'metrics': self.metrics,
                'clients': len(self.connected_clients),
                'last_classification': self.current_state.timestamp if self.current_state else None
            }
        
        @self.app.route('/api/mode', methods=['GET', 'POST'])
        def handle_mode():
            """Get or set server mode"""
            if request.method == 'GET':
                return {'mode': self.mode}
            else:
                data = request.json
                new_mode = data.get('mode')
                try:
                    self.set_mode(new_mode)
                    return {'success': True, 'mode': self.mode}
                except ValueError as e:
                    return {'error': str(e)}, 400
        
        @self.app.route('/api/current')
        def get_current_state():
            """Get current classification state"""
            if self.current_state:
                return asdict(self.current_state)
            else:
                return {'error': 'No current state available'}, 404
        
        @self.app.route('/api/history')
        def get_history():
            """Get recent classification history"""
            limit = request.args.get('limit', 50, type=int)
            limit = min(limit, len(self.classification_history))
            
            recent_history = self.classification_history[-limit:]
            return {
                'history': [asdict(result) for result in recent_history],
                'total_count': len(self.classification_history),
                'returned_count': len(recent_history)
            }
        
        @self.app.route('/api/clients')
        def get_clients():
            """Get information about connected clients"""
            with self.client_lock:
                client_info = []
                for session_id, client in self.connected_clients.items():
                    client_info.append({
                        'session_id': session_id,
                        'remote_addr': client.remote_addr,
                        'connected_at': client.connected_at,
                        'last_ping': client.last_ping,
                        'authenticated': client.authenticated,
                        'client_type': client.client_type,
                        'connection_duration': time.time() - client.connected_at
                    })
                
                return {
                    'clients': client_info,
                    'total_connected': len(client_info)
                }
        
        @self.app.route('/api/metrics')
        def get_metrics():
            """Get detailed server metrics"""
            uptime = time.time() - self.metrics['server_start_time']
            enhanced_metrics = self.metrics.copy()
            enhanced_metrics.update({
                'uptime': uptime,
                'messages_per_second': self.metrics['messages_sent'] / uptime if uptime > 0 else 0,
                'avg_clients': self.metrics['total_connections'] / uptime if uptime > 0 else 0,
                'classification_rate': self.metrics['classification_updates'] / uptime if uptime > 0 else 0
            })
            
            return enhanced_metrics
    
    def _setup_socket_events(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            session_id = request.sid
            remote_addr = request.remote_addr
            user_agent = request.headers.get('User-Agent', 'Unknown')
            
            # Create client info
            client_info = ClientInfo(
                session_id=session_id,
                remote_addr=remote_addr,
                user_agent=user_agent,
                connected_at=time.time(),
                last_ping=time.time()
            )
            
            # Store client info
            with self.client_lock:
                self.connected_clients[session_id] = client_info
                self.metrics['total_connections'] += 1
                self.metrics['current_connections'] = len(self.connected_clients)
            
            self.logger.info(f"Client connected: {session_id} from {remote_addr}")
            
            # Send welcome message with current state
            emit('connection_ack', {
                'session_id': session_id,
                'server_time': time.time(),
                'mode': self.mode,
                'message': f'Connected to Biometric Authentication Server ({self.mode} mode)'
            })
            
            # Send current state if available
            if self.current_state:
                emit('classification', asdict(self.current_state))
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            session_id = request.sid
            
            with self.client_lock:
                if session_id in self.connected_clients:
                    client_info = self.connected_clients[session_id]
                    connection_duration = time.time() - client_info.connected_at
                    del self.connected_clients[session_id]
                    self.metrics['current_connections'] = len(self.connected_clients)
                    
                    self.logger.info(f"Client disconnected: {session_id} "
                                   f"(duration: {connection_duration:.1f}s)")
        
        @self.socketio.on('ping')
        def handle_ping(data):
            """Handle client ping for connection keepalive"""
            session_id = request.sid
            
            with self.client_lock:
                if session_id in self.connected_clients:
                    self.connected_clients[session_id].last_ping = time.time()
            
            # Respond with pong
            emit('pong', {'server_time': time.time(), 'client_data': data})
        
        @self.socketio.on('authenticate')
        def handle_authenticate(data):
            """Handle client authentication for crypto wallet integration"""
            session_id = request.sid
            
            # TODO: Implement proper authentication logic for production
            # For now, accept any authentication attempt
            auth_token = data.get('token', '')
            client_type = data.get('client_type', 'unknown')
            
            with self.client_lock:
                if session_id in self.connected_clients:
                    self.connected_clients[session_id].authenticated = True
                    self.connected_clients[session_id].client_type = client_type
            
            self.logger.info(f"Client authenticated: {session_id} as {client_type}")
            
            emit('auth_response', {
                'authenticated': True,
                'client_type': client_type,
                'permissions': ['read_classifications', 'read_muscle_state']
            })
        
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            """Handle client subscription to specific data types"""
            session_id = request.sid
            data_types = data.get('data_types', ['classification', 'muscle_state'])
            
            # TODO: Implement subscription filtering
            emit('subscription_ack', {
                'subscribed_types': data_types,
                'message': 'Subscription confirmed'
            })
        
        @self.socketio.on('request_history')
        def handle_history_request(data):
            """Handle request for historical data"""
            limit = data.get('limit', 10)
            limit = min(limit, 100)  # Limit to prevent overwhelming clients
            
            recent_history = self.classification_history[-limit:]
            
            emit('history_data', {
                'history': [asdict(result) for result in recent_history],
                'count': len(recent_history)
            })
    
    def _get_next_sequence(self) -> int:
        """Get next sequence number for message ordering"""
        with self.sequence_lock:
            self.sequence_counter += 1
            return self.sequence_counter
    
    def process_real_classification(self, class_id: int, confidence: float, processing_time: float):
        """Process real EEG classification result"""
        hash_mode = self._convert_class_id_to_hash(class_id)
        
        classification_data = {
            'mode': hash_mode,
            'confidence': confidence,
            'processing_time': processing_time
        }
        
        muscle_data = {
            'binary': '0000',  # Real muscle data would come from hardware
            'decimal': 0,
            'active_muscles': []
        }
        
        system_data = {
            'server_mode': 'real',
            'eeg_quality': random.uniform(0.7, 0.9),
            'processing_latency': processing_time
        }
        
        self.broadcast_classification(classification_data, muscle_data, system_data)
        
        print(f"Real classification processed: Class ID {class_id} -> {CLASS_NAMES[class_id]} -> {hash_mode[:16]}... (confidence: {confidence:.2f})")
    
    def broadcast_classification(self, classification_data: Dict[str, Any], 
                               muscle_data: Dict[str, Any],
                               system_data: Dict[str, Any]):
        """
        Broadcast classification result to all connected clients
        
        Args:
            classification_data: EEG classification result with 'mode' hash
            muscle_data: Muscle sensor data
            system_data: System status data
        """
        try:
            # Create classification result
            result = ClassificationResult(
                timestamp=time.time(),
                sequence=self._get_next_sequence(),
                classification=classification_data,
                muscle_state=muscle_data,
                system=system_data
            )
            
            # Store current state and history
            self.current_state = result
            self.classification_history.append(result)
            
            # Limit history size
            if len(self.classification_history) > self.max_history_size:
                self.classification_history = self.classification_history[-self.max_history_size:]
            
            # Broadcast to all connected clients
            with self.client_lock:
                if self.connected_clients:
                    self.socketio.emit('classification', asdict(result))
                    self.metrics['messages_sent'] += len(self.connected_clients)
                    self.metrics['classification_updates'] += 1
            
        except Exception as e:
            self.logger.error(f"Error broadcasting classification: {e}")
            self.metrics['errors'] += 1
    
    def broadcast_muscle_state(self, muscle_data: Dict[str, Any]):
        """
        Broadcast muscle state update to all connected clients
        
        Args:
            muscle_data: Muscle sensor data
        """
        try:
            message = {
                'timestamp': time.time(),
                'sequence': self._get_next_sequence(),
                'muscle_state': muscle_data
            }
            
            with self.client_lock:
                if self.connected_clients:
                    self.socketio.emit('muscle_state', message)
                    self.metrics['messages_sent'] += len(self.connected_clients)
                    self.metrics['muscle_updates'] += 1
            
        except Exception as e:
            self.logger.error(f"Error broadcasting muscle state: {e}")
            self.metrics['errors'] += 1
    
    def broadcast_system_status(self, status_data: Dict[str, Any]):
        """
        Broadcast system status to all connected clients
        
        Args:
            status_data: System status information
        """
        try:
            message = {
                'timestamp': time.time(),
                'sequence': self._get_next_sequence(),
                'status': status_data
            }
            
            with self.client_lock:
                if self.connected_clients:
                    self.socketio.emit('system_status', message)
                    self.metrics['messages_sent'] += len(self.connected_clients)
            
        except Exception as e:
            self.logger.error(f"Error broadcasting system status: {e}")
            self.metrics['errors'] += 1
    
    def broadcast_error(self, error_message: str, error_type: str = "general"):
        """
        Broadcast error message to all connected clients
        
        Args:
            error_message: Error description
            error_type: Type of error
        """
        try:
            message = {
                'timestamp': time.time(),
                'sequence': self._get_next_sequence(),
                'error': {
                    'message': error_message,
                    'type': error_type
                }
            }
            
            with self.client_lock:
                if self.connected_clients:
                    self.socketio.emit('error', message)
                    self.metrics['messages_sent'] += len(self.connected_clients)
                    self.metrics['errors'] += 1
            
        except Exception as e:
            self.logger.error(f"Error broadcasting error message: {e}")
    
    def cleanup_stale_connections(self):
        """Remove stale client connections"""
        current_time = time.time()
        stale_timeout = 300  # 5 minutes
        
        with self.client_lock:
            stale_clients = []
            for session_id, client in self.connected_clients.items():
                if current_time - client.last_ping > stale_timeout:
                    stale_clients.append(session_id)
            
            for session_id in stale_clients:
                del self.connected_clients[session_id]
                self.logger.info(f"Removed stale client: {session_id}")
            
            if stale_clients:
                self.metrics['current_connections'] = len(self.connected_clients)
    
    def get_client_count(self) -> int:
        """Get number of connected clients"""
        with self.client_lock:
            return len(self.connected_clients)
    
    def run(self, threaded=True):
        """
        Start the WebSocket server
        
        Args:
            threaded: Whether to run in threaded mode
        """
        self.logger.info(f"Starting WebSocket server on {self.host}:{self.port} in {self.mode} mode")
        
        try:
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=self.debug,
                use_reloader=False,
                allow_unsafe_werkzeug=True
            )
        except Exception as e:
            self.logger.error(f"Server error: {e}")
            raise

# Demo/Test Functions
def demo_websocket_server():
    """Demo function to test WebSocket server with simulated data"""
    import threading
    import time
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create server in demo mode
    server = BiometricWebSocketServer(host='192.168.229.84', port=5000, debug=True, mode='demo')
    
    # Start server in a separate thread
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(2)
    
    print("WebSocket server started in DEMO mode")
    print(f"Access at http://{server.host}:{server.port}")
    print(f"WebSocket endpoint: ws://{server.host}:{server.port}")
    print("Sending simulated classification data...")
    
    # Simulate data broadcasting
    try:
        for i in range(100):
            # Simulate classification data with hash modes
            class_id = random.randint(0, 2)
            hash_mode = server._convert_class_id_to_hash(class_id)
            
            classification_data = {
                'mode': hash_mode,
                'confidence': random.uniform(0.7, 0.95),
                'processing_time': random.uniform(20, 80)
            }
            
            # Simulate muscle data
            muscle_data = {
                'binary': f"{random.randint(0, 15):04b}",
                'decimal': random.randint(0, 15),
                'active_muscles': random.sample(['RightPec', 'LeftPec', 'RightQuad', 'LeftLeg'], 
                                              random.randint(0, 2))
            }
            
            # Simulate system data
            system_data = {
                'server_mode': 'demo',
                'eeg_quality': random.uniform(0.6, 0.9),
                'processing_latency': random.uniform(30, 100)
            }
            
            # Broadcast data
            server.broadcast_classification(classification_data, muscle_data, system_data)
            
            print(f"Sent classification {i+1}: {CLASS_NAMES[class_id]} "
                  f"(confidence: {classification_data['confidence']:.2f}) "
                  f"-> {hash_mode[:16]}...")
            
            time.sleep(1)  # Send every second
            
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    
    print("Demo completed. Server continues running...")
    print("Press Ctrl+C to stop the server")
    
    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")

def real_websocket_server(eeg_classifier=None):
    """Real mode WebSocket server"""
    import threading
    import time
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create server in real mode
    server = BiometricWebSocketServer(host='192.168.229.84', port=5000, debug=True, mode='real')
    
    # Set EEG classifier if provided
    if eeg_classifier:
        server.set_eeg_classifier(eeg_classifier)
    
    print("Starting WebSocket server in REAL mode...")
    print(f"Access at http://{server.host}:{server.port}")
    print(f"WebSocket endpoint: ws://{server.host}:{server.port}")
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")

if __name__ == '__main__':
    import sys
    
    # Check command line arguments for mode selection
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == 'demo':
            demo_websocket_server()
        elif mode == 'real':
            real_websocket_server()
        else:
            print("Usage: python websocket_server.py [demo|real]")
            print("Default: demo mode")
            demo_websocket_server()
    else:
        # Default to demo mode
        demo_websocket_server() 