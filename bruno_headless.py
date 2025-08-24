#!/usr/bin/env python3
"""
Bruno Headless Mode - No Display Required
Runs bottle detection and approach without GUI windows
Perfect for SSH connections and headless operation
"""

import os
import sys
import json
import time
import logging
import signal
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set environment to prevent Qt issues
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Import OpenCV after setting environment
import cv2
import numpy as np

from bottle_detection.bruno_bottle_detector import BrunoBottleDetector

class BrunoHeadless(BrunoBottleDetector):
    def __init__(self, config_file: str = None):
        # Initialize parent class
        super().__init__(config_file)
        
        # Headless specific settings
        self.save_detections = True
        self.detection_images_dir = "detections"
        self.log_interval = 5  # seconds
        self.last_log_time = 0
        
        # Create detection images directory
        if self.save_detections:
            os.makedirs(self.detection_images_dir, exist_ok=True)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info("Bruno running in HEADLESS mode - no display required")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum} - shutting down gracefully")
        self.running = False
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame without displaying (headless mode)"""
        # Call parent process_frame but don't display
        processed_frame = super().process_frame(frame)
        
        # Save detection images if enabled
        if self.save_detections and self.last_bottles:
            self.save_detection_image(processed_frame)
        
        # Log status periodically
        self.log_status_periodically()
        
        return processed_frame
    
    def save_detection_image(self, frame: np.ndarray):
        """Save images when bottles are detected"""
        if not self.last_bottles:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bottle_count = len(self.last_bottles)
        
        # Get distance info if available
        distance_info = ""
        if self.last_movement_command:
            distance_cm = self.last_movement_command.get('distance_cm', 0)
            action = self.last_movement_command.get('action', 'UNKNOWN')
            distance_info = f"_{distance_cm:.0f}cm_{action}"
        
        filename = f"{self.detection_images_dir}/bottle_{timestamp}_{bottle_count}bottles{distance_info}.jpg"
        
        try:
            cv2.imwrite(filename, frame)
            self.logger.debug(f"Saved detection: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save detection image: {e}")
    
    def log_status_periodically(self):
        """Log status information periodically"""
        current_time = time.time()
        
        if current_time - self.last_log_time >= self.log_interval:
            self.last_log_time = current_time
            
            # Log current status
            bottle_count = len(self.last_bottles) if self.last_bottles else 0
            status_info = [
                f"🤖 Bruno Status: {'APPROACHING' if self.approach_mode else 'SCANNING'}",
                f"📱 Bottles detected: {bottle_count}",
                f"📊 Total detections: {self.detection_count}",
                f"🎯 Bottles reached: {self.bottles_reached}",
                f"🔄 Movement: {'ACTIVE' if self.movement_controller.is_moving else 'STOPPED'}"
            ]
            
            # Add distance info if available
            if self.last_movement_command:
                distance_cm = self.last_movement_command.get('distance_cm', 0)
                action = self.last_movement_command.get('action', 'IDLE')
                zone = self.last_movement_command.get('distance_zone', 'N/A')
                status_info.append(f"📏 Distance: {distance_cm:.1f}cm ({zone}) - Action: {action}")
            
            # Log all status info
            for info in status_info:
                self.logger.info(info)
            
            self.logger.info("-" * 50)
    
    def run(self):
        """Main execution loop for headless mode"""
        self.logger.info("🚀 Starting Bruno Headless Bottle Detection & Approach System")
        self.logger.info("📱 No display required - perfect for SSH connections")
        self.logger.info("🎯 Bruno will detect bottles and approach them, stopping at 1 foot")
        self.logger.info("⏹️  Press Ctrl+C to stop gracefully")
        
        if self.save_detections:
            self.logger.info(f"💾 Detection images will be saved to: {self.detection_images_dir}/")
        
        self.running = True
        
        try:
            frame_count = 0
            
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    self.logger.error("Failed to capture frame")
                    time.sleep(1)  # Wait before retrying
                    continue
                
                frame_count += 1
                
                # Process frame (no display)
                self.process_frame(frame)
                
                # Brief pause to prevent excessive CPU usage
                time.sleep(0.05)  # ~20 FPS processing
                
                # Log frame count occasionally
                if frame_count % 600 == 0:  # Every ~30 seconds at 20 FPS
                    self.logger.info(f"📹 Processed {frame_count} frames")
        
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Enhanced cleanup for headless mode"""
        self.logger.info("🛑 Shutting down Bruno...")
        
        # Call parent cleanup
        super().cleanup()
        
        # Headless specific cleanup
        if self.save_detections:
            detection_files = len([f for f in os.listdir(self.detection_images_dir) 
                                 if f.endswith('.jpg')])
            self.logger.info(f"💾 Saved {detection_files} detection images")
        
        # Final status summary
        self.logger.info("📊 Final Statistics:")
        self.logger.info(f"   🔍 Total detections: {self.detection_count}")
        self.logger.info(f"   🎯 Bottles reached: {self.bottles_reached}")
        self.logger.info(f"   📱 Bottles found total: {self.bottles_found_total}")
        
        self.logger.info("✅ Bruno shutdown complete")

def main():
    """Main function for headless operation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bruno Headless - Bottle Detection Robot')
    parser.add_argument('--config', '-c', 
                       default='config/bruno_config.json',
                       help='Configuration file path')
    parser.add_argument('--log-level', '-l',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level')
    parser.add_argument('--no-save', action='store_true',
                       help='Disable saving detection images')
    parser.add_argument('--log-interval', type=int, default=5,
                       help='Status log interval in seconds')
    
    args = parser.parse_args()
    
    # Setup logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('bruno_headless.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("🤖 BRUNO HEADLESS MODE STARTING")
    logger.info("=" * 60)
    
    try:
        # Create Bruno instance
        bruno = BrunoHeadless(args.config)
        
        # Apply command line options
        if args.no_save:
            bruno.save_detections = False
            logger.info("💾 Detection image saving disabled")
        
        bruno.log_interval = args.log_interval
        logger.info(f"📊 Status logging every {args.log_interval} seconds")
        
        # Run Bruno
        bruno.run()
        
    except KeyboardInterrupt:
        logger.info("🛑 Program interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("👋 Bruno Headless Mode Ended")

if __name__ == "__main__":
    main()