#!/usr/bin/env python3
"""
Test Movement System
Verifies that Bruno can move in different patterns
"""

import os
import sys
import time
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from robot_control.roomba_navigator import RoombaNavigator

def test_navigation_patterns():
    """Test different navigation patterns"""
    print("Testing Bruno Navigation Patterns")
    print("=" * 40)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize navigator
        navigator = RoombaNavigator()
        print(f"Navigator initialized")
        print(f"Hardware available: {'Yes' if navigator.hardware_available else 'No (Simulation)'}")
        
        # Test individual movements
        test_movements = [
            ("Forward", lambda: navigator.move_forward()),
            ("Turn Left", lambda: navigator.turn_left()),
            ("Turn Right", lambda: navigator.turn_right()),
            ("Backward", lambda: navigator.move_backward()),
            ("Arc Left", lambda: navigator.arc_turn(20, 15, "left")),
            ("Arc Right", lambda: navigator.arc_turn(20, 15, "right")),
        ]
        
        print(f"\nTesting Individual Movements:")
        for name, movement_func in test_movements:
            print(f"   Testing {name}...", end=" ")
            try:
                movement_func()
                time.sleep(2)  # Move for 2 seconds
                navigator.stop()
                time.sleep(1)  # Pause between movements
                print("PASS")
            except Exception as e:
                print(f"ERROR: {e}")
        
        # Test navigation state machine
        print(f"\nTesting Navigation State Machine:")
        print("Running autonomous navigation for 30 seconds...")
        
        start_time = time.time()
        test_duration = 30  # seconds
        
        while time.time() - start_time < test_duration:
            # Simulate bottle detection occasionally
            bottle_detected = (int(time.time()) % 10 == 0)  # Every 10 seconds
            bottle_position = (320, 240) if bottle_detected else None
            
            # Update navigation
            nav_command = navigator.update_navigation(
                bottle_detected=bottle_detected,
                bottle_position=bottle_position
            )
            
            # Log navigation state
            if int(time.time()) % 3 == 0:  # Every 3 seconds
                status = navigator.get_status()
                elapsed = time.time() - start_time
                print(f"   {elapsed:05.1f}s: {status['state']} - {nav_command.get('action', 'none')}")
            
            time.sleep(0.1)  # 10Hz update rate
        
        print("\nNavigation Test Results:")
        final_status = navigator.get_status()
        print(f"   Final state: {final_status['state']}")
        print(f"   Hardware working: {'YES' if final_status['hardware_available'] else 'Simulated'}")
        print(f"   Stuck count: {final_status['stuck_count']}")
        
        return True
        
    except Exception as e:
        print(f"Navigation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        try:
            navigator.cleanup()
        except:
            pass

def test_bottle_approach_simulation():
    """Test bottle approach behavior without camera"""
    print(f"\nTesting Bottle Approach Simulation:")
    
    try:
        navigator = RoombaNavigator()
        
        # Simulate bottles at different positions
        bottle_positions = [
            (160, 240, "Left side"),      # Left of center
            (480, 240, "Right side"),     # Right of center  
            (320, 240, "Center"),         # Center
            (100, 240, "Far left"),       # Far left
            (540, 240, "Far right"),      # Far right
        ]
        
        for x, y, description in bottle_positions:
            print(f"   Testing approach to {description} bottle at ({x}, {y})...")
            
            # Test approach for 5 seconds
            for i in range(5):
                nav_command = navigator.update_navigation(
                    bottle_detected=True,
                    bottle_position=(x, y)
                )
                
                action = nav_command.get('action', 'unknown')
                print(f"     Step {i+1}: {action}")
                time.sleep(1)
            
            navigator.stop()
            time.sleep(1)
        
        print("Bottle approach simulation completed")
        return True
        
    except Exception as e:
        print(f"Bottle approach test failed: {e}")
        return False
    
    finally:
        try:
            navigator.cleanup()
        except:
            pass

def main():
    """Main test function"""
    print("Bruno Movement Test Suite")
    print("=" * 50)
    
    # Test 1: Navigation patterns
    nav_test_passed = test_navigation_patterns()
    
    # Test 2: Bottle approach simulation
    approach_test_passed = test_bottle_approach_simulation()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    print(f"   Navigation patterns: {'PASSED' if nav_test_passed else 'FAILED'}")
    print(f"   Bottle approach: {'PASSED' if approach_test_passed else 'FAILED'}")
    
    if nav_test_passed and approach_test_passed:
        print("\nALL TESTS PASSED!")
        print("Bruno is ready for Roomba mode:")
        print("   python bruno_roomba.py")
    else:
        print("\nSome tests failed")
        print("Check hardware connections and try:")
        print("   python bruno_simple.py  # Simpler version")

if __name__ == "__main__":
    main()