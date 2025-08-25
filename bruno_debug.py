#!/usr/bin/env python3
"""
Bruno Debug Mode - Shows detailed information about detection and movement
"""

import os
import sys
import time
import logging

# Set environment
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Debug main function with detailed logging"""
    
    # Setup verbose logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('bruno_debug.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    print("🐛 Bruno Debug Mode")
    print("=" * 30)
    
    # Test imports
    logger.info("Testing imports...")
    try:
        from bruno_simple import BrunoSimple
        logger.info("✅ BrunoSimple imported successfully")
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return
    
    # Test configuration loading
    logger.info("Testing configuration...")
    try:
        bruno = BrunoSimple("config/bruno_config.json")
        logger.info("✅ Bruno instance created")
        
        # Show configuration
        print("\n📋 Configuration Summary:")
        print(f"   Camera: {bruno.config['camera']['device_id']}")
        print(f"   Detection confidence: {bruno.config['detection']['confidence_threshold']}")
        print(f"   Stop distance: {bruno.config['distance_estimation']['stop_distance']}cm")
        print(f"   Approach enabled: {bruno.config.get('behavior', {}).get('approach_bottles_automatically', False)}")
        
        # Show controller status
        print(f"   Head controller: {'✅ OK' if bruno.head_controller else '❌ Failed'}")
        print(f"   Movement controller: {'✅ OK' if bruno.movement_controller else '❌ Failed'}")
        
    except Exception as e:
        logger.error(f"❌ Configuration failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test camera
    logger.info("Testing camera connection...")
    try:
        ret, frame = bruno.camera.read()
        if ret:
            logger.info(f"✅ Camera working - Frame size: {frame.shape}")
        else:
            logger.error("❌ Camera not providing frames")
            return
    except Exception as e:
        logger.error(f"❌ Camera test failed: {e}")
        return
    
    # Test detection system
    logger.info("Testing bottle detection...")
    try:
        bottles, annotated_frame = bruno.bottle_detector.detect_bottles(frame)
        logger.info(f"✅ Detection system working - Found {len(bottles)} bottles")
        
        if bottles:
            best_bottle = bruno.bottle_detector.get_best_bottle(bottles)
            logger.info(f"   Best bottle confidence: {best_bottle['confidence']:.2f}")
            
            # Test distance estimation
            frame_height, frame_width = frame.shape[:2]
            movement_cmd = bruno.distance_estimator.get_movement_command(best_bottle, frame_width, frame_height)
            logger.info(f"   Distance: {movement_cmd['distance_cm']:.1f}cm")
            logger.info(f"   Action: {movement_cmd['action']}")
            
    except Exception as e:
        logger.error(f"❌ Detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🚀 Running Bruno with debug logging...")
    print("📝 Check bruno_debug.log for detailed logs")
    print("⏹️  Press Ctrl+C to stop")
    
    # Run Bruno with debug logging
    try:
        bruno.run()
    except KeyboardInterrupt:
        logger.info("Debug session stopped by user")
    except Exception as e:
        logger.error(f"Runtime error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()