#!/usr/bin/env python3
"""
Test script for Bruno GPT Vision integration
Tests the basic functionality without requiring actual hardware or API keys.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_config_loading():
    """Test that configuration can be loaded"""
    print("Testing configuration loading...")
    try:
        with open("config/bruno_config.json", 'r') as f:
            config = json.load(f)
        print("✓ Configuration loaded successfully")
        return config
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return None

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing module imports...")
    
    # Test OpenAI import
    try:
        from openai import OpenAI
        print("✓ OpenAI library available")
    except ImportError:
        print("⚠ OpenAI library not available (will use local detection)")
    
    # Test Bruno modules
    try:
        from src.robot_control.movement_controller import MovementController
        from src.robot_control.head_controller import HeadController
        from src.bottle_detection.bottle_detector import BottleDetector
        print("✓ Bruno modules available")
        return True
    except ImportError as e:
        print(f"⚠ Bruno modules not available: {e}")
        return False

def test_gpt_detector_creation():
    """Test GPT detector creation"""
    print("Testing GPT detector creation...")
    try:
        from gpt import GPTDetector
        
        config = {
            'detection': {
                'confidence_threshold': 0.6
            }
        }
        
        detector = GPTDetector(config)
        print("✓ GPT detector created successfully")
        return True
    except Exception as e:
        print(f"✗ GPT detector creation failed: {e}")
        return False

def test_camera_creation():
    """Test camera interface creation"""
    print("Testing camera interface creation...")
    try:
        from gpt import BrunoCamera
        
        config = {
            'device_id': 0,
            'width': 640,
            'height': 480,
            'fps': 30,
            'flip_horizontal': False
        }
        
        camera = BrunoCamera(config)
        print("✓ Camera interface created successfully")
        return True
    except Exception as e:
        print(f"✗ Camera interface creation failed: {e}")
        return False

def test_robot_controller_creation():
    """Test robot controller creation"""
    print("Testing robot controller creation...")
    try:
        from gpt import BrunoRobotController
        
        config = {
            'movement_control': {
                'max_speed': 40,
                'min_speed': 15
            },
            'head_control': {
                'enabled': True
            },
            'collision_avoidance': {
                'safe_distance': 100,
                'caution_distance': 150,
                'danger_distance': 80
            }
        }
        
        robot = BrunoRobotController(config, dry_run=True)
        print("✓ Robot controller created successfully")
        return True
    except Exception as e:
        print(f"✗ Robot controller creation failed: {e}")
        return False

def test_collision_avoidance():
    """Test collision avoidance system"""
    print("Testing collision avoidance system...")
    try:
        from gpt import CollisionAvoidance, SafetyLevel
        
        config = {
            'collision_avoidance': {
                'safe_distance': 100,
                'caution_distance': 150,
                'danger_distance': 80
            }
        }
        
        collision_system = CollisionAvoidance(config)
        print("✓ Collision avoidance system created successfully")
        
        # Test safety level determination
        from gpt import ObstacleInfo
        obstacles = [
            ObstacleInfo(distance=50, angle=0, confidence=0.8, type="wall"),
            ObstacleInfo(distance=200, angle=30, confidence=0.6, type="object")
        ]
        
        safety_level = collision_system.check_safety(obstacles)
        print(f"✓ Safety level determination works: {safety_level}")
        
        return True
    except Exception as e:
        print(f"✗ Collision avoidance test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Bruno GPT Vision Integration Test")
    print("=" * 40)
    
    # Test configuration
    config = test_config_loading()
    if not config:
        print("Cannot proceed without configuration")
        return False
    
    # Test imports
    bruno_available = test_imports()
    
    # Test component creation
    tests = [
        test_gpt_detector_creation,
        test_camera_creation,
        test_robot_controller_creation,
        test_collision_avoidance
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("✓ All tests passed! GPT integration is ready.")
        return True
    else:
        print("⚠ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
