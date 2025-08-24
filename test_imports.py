#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test importing all Bruno modules"""
    print("🧪 Testing Bruno module imports...")
    
    try:
        print("   Testing bottle_detector...", end=" ")
        from bottle_detection.bottle_detector import BottleDetector
        print("✅")
        
        print("   Testing distance_estimator...", end=" ")
        from bottle_detection.distance_estimator import DistanceEstimator  
        print("✅")
        
        print("   Testing head_controller...", end=" ")
        from robot_control.head_controller import HeadController
        print("✅")
        
        print("   Testing movement_controller...", end=" ")
        from robot_control.movement_controller import MovementController
        print("✅")
        
        print("   Testing bruno_bottle_detector...", end=" ")
        from bottle_detection.bruno_bottle_detector import BrunoBottleDetector
        print("✅")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def test_create_instances():
    """Test creating instances without hardware"""
    print("\n🔧 Testing instance creation...")
    
    try:
        from bottle_detection.bottle_detector import BottleDetector
        from bottle_detection.distance_estimator import DistanceEstimator
        
        print("   Creating BottleDetector...", end=" ")
        detector = BottleDetector()
        print("✅")
        
        print("   Creating DistanceEstimator...", end=" ")
        estimator = DistanceEstimator()
        print("✅")
        
        print("\n🎉 Instance creation successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Instance creation failed: {e}")
        return False

def main():
    print("🤖 Bruno Import Test")
    print("=" * 30)
    
    imports_ok = test_imports()
    instances_ok = test_create_instances()
    
    print("\n" + "=" * 30)
    if imports_ok and instances_ok:
        print("✅ ALL TESTS PASSED")
        print("🚀 Bruno is ready to run!")
        print("\nNext steps:")
        print("   python bruno_headless.py     # Headless mode")
        print("   python web_interface.py      # Web interface")  
        print("   python start_bruno.py        # Auto-detect mode")
    else:
        print("❌ Some tests failed")
        print("🔧 Check the error messages above")

if __name__ == "__main__":
    main()