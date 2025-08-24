#!/usr/bin/env python3
"""
Web Interface for Bruno Bottle Detection Robot
Provides remote monitoring and control via web browser
Headless operation - no Qt/display required
"""

import os
import sys
import json
import time
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify, request, Response

# Set environment to prevent Qt issues
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import cv2
import base64
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from bruno_headless import BrunoHeadless

class BrunoWebInterface:
    def __init__(self, config_file: str = None):
        self.app = Flask(__name__)
        self.bruno = None
        self.config_file = config_file or "config/bruno_config.json"
        
        # Status tracking
        self.status = {
            'running': False,
            'bottles_detected': 0,
            'current_action': 'IDLE',
            'distance_to_bottle': 0,
            'last_update': None,
            'error': None
        }
        
        # Setup routes
        self.setup_routes()
        
        # Frame storage for web streaming
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('bruno_control.html')
        
        @self.app.route('/api/status')
        def api_status():
            """Get current Bruno status"""
            return jsonify(self.status)
        
        @self.app.route('/api/start', methods=['POST'])
        def api_start():
            """Start Bruno detection system"""
            try:
                if not self.bruno:
                    self.bruno = BrunoHeadless(self.config_file)
                
                if not self.status['running']:
                    # Start Bruno in separate thread
                    self.bruno_thread = threading.Thread(target=self.run_bruno, daemon=True)
                    self.bruno_thread.start()
                    self.status['running'] = True
                    self.status['error'] = None
                
                return jsonify({'success': True, 'message': 'Bruno started'})
                
            except Exception as e:
                self.status['error'] = str(e)
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/stop', methods=['POST'])
        def api_stop():
            """Stop Bruno detection system"""
            try:
                self.status['running'] = False
                if self.bruno:
                    self.bruno.running = False
                
                return jsonify({'success': True, 'message': 'Bruno stopped'})
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/emergency_stop', methods=['POST'])
        def api_emergency_stop():
            """Emergency stop all movement"""
            try:
                if self.bruno and hasattr(self.bruno, 'movement_controller'):
                    self.bruno.movement_controller.emergency_stop()
                
                self.status['current_action'] = 'EMERGENCY_STOP'
                return jsonify({'success': True, 'message': 'Emergency stop activated'})
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/config', methods=['GET', 'POST'])
        def api_config():
            """Get or update configuration"""
            if request.method == 'GET':
                try:
                    with open(self.config_file, 'r') as f:
                        config = json.load(f)
                    return jsonify(config)
                except Exception as e:
                    return jsonify({'error': str(e)})
            
            elif request.method == 'POST':
                try:
                    new_config = request.json
                    with open(self.config_file, 'w') as f:
                        json.dump(new_config, f, indent=2)
                    return jsonify({'success': True, 'message': 'Configuration updated'})
                except Exception as e:
                    return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/video_feed')
        def video_feed():
            """Video streaming route"""
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/api/calibrate/<component>')
        def api_calibrate(component):
            """Calibrate different components"""
            try:
                if component == 'head' and self.bruno:
                    self.bruno.head_controller.calibrate_servo()
                    return jsonify({'success': True, 'message': f'{component} calibrated'})
                elif component == 'movement' and self.bruno:
                    self.bruno.movement_controller.calibrate_speeds()
                    return jsonify({'success': True, 'message': f'{component} calibrated'})
                else:
                    return jsonify({'success': False, 'error': f'Unknown component: {component}'})
                    
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
    
    def run_bruno(self):
        """Run Bruno in separate thread with status updates"""
        try:
            # Override Bruno's process_frame method to capture frames for web
            original_process_frame = self.bruno.process_frame
            
            def web_process_frame(frame):
                processed_frame = original_process_frame(frame)
                
                # Store frame for web streaming
                with self.frame_lock:
                    self.latest_frame = processed_frame.copy()
                
                # Update status from Bruno's state
                self.update_status()
                
                return processed_frame
            
            self.bruno.process_frame = web_process_frame
            self.bruno.run()
            
        except Exception as e:
            self.status['error'] = str(e)
            self.status['running'] = False
    
    def update_status(self):
        """Update status from Bruno's current state"""
        if not self.bruno:
            return
        
        self.status.update({
            'bottles_detected': len(getattr(self.bruno, 'last_bottles', [])),
            'last_update': datetime.now().isoformat(),
            'running': self.bruno.running if hasattr(self.bruno, 'running') else False
        })
        
        # Get movement status if available
        if hasattr(self.bruno, 'movement_controller'):
            movement_status = self.bruno.movement_controller.get_status()
            if movement_status['current_command']:
                cmd = movement_status['current_command']
                self.status['current_action'] = cmd.get('action', 'UNKNOWN')
                self.status['distance_to_bottle'] = cmd.get('distance_cm', 0)
        
        # Get detection count
        self.status['total_detections'] = getattr(self.bruno, 'detection_count', 0)
    
    def generate_frames(self):
        """Generate frames for video streaming"""
        while True:
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()
                else:
                    # Create a placeholder frame
                    frame = self.create_placeholder_frame()
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.1)  # ~10 FPS for web streaming
    
    def create_placeholder_frame(self):
        """Create a placeholder frame when no camera feed is available"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        text = "Bruno Camera Feed"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2)
        
        status_text = f"Status: {'Running' if self.status['running'] else 'Stopped'}"
        cv2.putText(frame, status_text, (10, 30), font, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the web interface"""
        print(f"Starting Bruno Web Interface on http://{host}:{port}")
        print(f"Access Bruno control panel at: http://{host}:{port}")
        
        # Create templates directory if it doesn't exist
        os.makedirs('templates', exist_ok=True)
        self.create_html_template()
        
        self.app.run(host=host, port=port, debug=debug, threaded=True)
    
    def create_html_template(self):
        """Create HTML template for Bruno control panel"""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Bruno - Bottle Detection Robot Control</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; }
        .panel { background: white; padding: 20px; margin: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { display: flex; gap: 20px; flex-wrap: wrap; }
        .status-item { flex: 1; min-width: 200px; text-align: center; }
        .status-value { font-size: 2em; font-weight: bold; color: #333; }
        .status-label { color: #666; }
        .controls button { margin: 5px; padding: 10px 20px; font-size: 16px; border: none; border-radius: 4px; cursor: pointer; }
        .start-btn { background: #4CAF50; color: white; }
        .stop-btn { background: #f44336; color: white; }
        .emergency-btn { background: #FF5722; color: white; }
        .calibrate-btn { background: #2196F3; color: white; }
        .video-container { text-align: center; }
        .video-feed { max-width: 100%; border-radius: 8px; }
        .running { color: #4CAF50; }
        .stopped { color: #f44336; }
        .error { color: #f44336; background: #ffebee; padding: 10px; border-radius: 4px; margin: 10px 0; }
        .log { height: 200px; overflow-y: scroll; background: #333; color: #0f0; font-family: monospace; padding: 10px; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Bruno - Bottle Detection Robot</h1>
        
        <div class="panel">
            <h2>Status Dashboard</h2>
            <div class="status">
                <div class="status-item">
                    <div class="status-value" id="status">Stopped</div>
                    <div class="status-label">System Status</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="bottles">0</div>
                    <div class="status-label">Bottles Detected</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="action">IDLE</div>
                    <div class="status-label">Current Action</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="distance">0</div>
                    <div class="status-label">Distance (cm)</div>
                </div>
            </div>
            <div id="error-display"></div>
        </div>
        
        <div class="panel">
            <h2>Controls</h2>
            <div class="controls">
                <button class="start-btn" onclick="startBruno()">Start Detection</button>
                <button class="stop-btn" onclick="stopBruno()">Stop</button>
                <button class="emergency-btn" onclick="emergencyStop()">EMERGENCY STOP</button>
                <button class="calibrate-btn" onclick="calibrateHead()">Calibrate Head</button>
                <button class="calibrate-btn" onclick="calibrateMovement()">Calibrate Movement</button>
            </div>
        </div>
        
        <div class="panel">
            <h2>Live Camera Feed</h2>
            <div class="video-container">
                <img class="video-feed" src="/video_feed" alt="Bruno Camera Feed">
            </div>
        </div>
        
        <div class="panel">
            <h2>Activity Log</h2>
            <div class="log" id="activity-log">
                Bruno Web Interface Ready...<br>
            </div>
        </div>
    </div>
    
    <script>
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status').textContent = data.running ? 'Running' : 'Stopped';
                    document.getElementById('status').className = data.running ? 'status-value running' : 'status-value stopped';
                    
                    document.getElementById('bottles').textContent = data.bottles_detected || 0;
                    document.getElementById('action').textContent = data.current_action || 'IDLE';
                    document.getElementById('distance').textContent = Math.round(data.distance_to_bottle || 0);
                    
                    // Show errors
                    const errorDiv = document.getElementById('error-display');
                    if (data.error) {
                        errorDiv.innerHTML = '<div class="error">Error: ' + data.error + '</div>';
                    } else {
                        errorDiv.innerHTML = '';
                    }
                })
                .catch(error => console.error('Status update error:', error));
        }
        
        function logActivity(message) {
            const log = document.getElementById('activity-log');
            const timestamp = new Date().toLocaleTimeString();
            log.innerHTML += timestamp + ' - ' + message + '<br>';
            log.scrollTop = log.scrollHeight;
        }
        
        function startBruno() {
            fetch('/api/start', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        logActivity('Bruno detection started');
                    } else {
                        logActivity('Failed to start: ' + data.error);
                    }
                });
        }
        
        function stopBruno() {
            fetch('/api/stop', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        logActivity('Bruno stopped');
                    } else {
                        logActivity('Failed to stop: ' + data.error);
                    }
                });
        }
        
        function emergencyStop() {
            fetch('/api/emergency_stop', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        logActivity('EMERGENCY STOP activated');
                    } else {
                        logActivity('Emergency stop failed: ' + data.error);
                    }
                });
        }
        
        function calibrateHead() {
            fetch('/api/calibrate/head')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        logActivity('Head calibration completed');
                    } else {
                        logActivity('Head calibration failed: ' + data.error);
                    }
                });
        }
        
        function calibrateMovement() {
            fetch('/api/calibrate/movement')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        logActivity('Movement calibration completed');
                    } else {
                        logActivity('Movement calibration failed: ' + data.error);
                    }
                });
        }
        
        // Update status every 2 seconds
        setInterval(updateStatus, 2000);
        updateStatus(); // Initial update
    </script>
</body>
</html>
        """
        
        with open('templates/bruno_control.html', 'w') as f:
            f.write(html_content)

def main():
    """Main function to run web interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bruno Web Interface')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--config', default='config/bruno_config.json', help='Config file')
    
    args = parser.parse_args()
    
    try:
        web_interface = BrunoWebInterface(args.config)
        web_interface.run(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\nWeb interface stopped")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Add numpy import for placeholder frame
    import numpy as np
    main()