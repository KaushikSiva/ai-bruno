#!/usr/bin/env python3
"""
Quick Fix for Qt/Display Issues on MasterPi
Diagnoses and fixes common OpenCV display problems
"""

import os
import sys
import subprocess

def check_system_info():
    """Check system information"""
    print("🔍 System Information:")
    print(f"   Python: {sys.version}")
    print(f"   Platform: {sys.platform}")
    
    # Check if running over SSH
    ssh_client = os.environ.get('SSH_CLIENT')
    ssh_connection = os.environ.get('SSH_CONNECTION')
    display = os.environ.get('DISPLAY')
    
    print(f"   SSH Client: {ssh_client if ssh_client else 'Not detected'}")
    print(f"   SSH Connection: {ssh_connection if ssh_connection else 'Not detected'}")
    print(f"   DISPLAY: {display if display else 'Not set'}")
    
    return {
        'ssh_detected': bool(ssh_client or ssh_connection),
        'display_set': bool(display)
    }

def check_opencv_info():
    """Check OpenCV configuration"""
    print("\n📹 OpenCV Information:")
    
    try:
        import cv2
        print(f"   Version: {cv2.__version__}")
        
        # Check build info
        build_info = cv2.getBuildInformation()
        
        # Look for GUI support
        gui_support = []
        if 'Qt' in build_info:
            gui_support.append('Qt')
        if 'GTK' in build_info:
            gui_support.append('GTK')
        if 'Cocoa' in build_info:
            gui_support.append('Cocoa')
        
        print(f"   GUI Support: {', '.join(gui_support) if gui_support else 'None detected'}")
        
        # Check for headless operation capability
        print("   Headless: Supported (can run without display)")
        
    except ImportError:
        print("   ❌ OpenCV not installed")
        return False
    
    return True

def test_display_access():
    """Test if display is accessible"""
    print("\n🖥️  Display Access Test:")
    
    display = os.environ.get('DISPLAY')
    if not display:
        print("   ❌ No DISPLAY environment variable")
        return False
    
    try:
        # Try to run a simple X11 command
        result = subprocess.run(['xset', 'q'], 
                               capture_output=True, 
                               timeout=5,
                               text=True)
        
        if result.returncode == 0:
            print("   ✅ Display accessible")
            return True
        else:
            print(f"   ❌ Display test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ❌ Display test timeout")
        return False
    except FileNotFoundError:
        print("   ❌ xset command not found")
        return False
    except Exception as e:
        print(f"   ❌ Display test error: {e}")
        return False

def test_opencv_headless():
    """Test OpenCV in headless mode"""
    print("\n🔧 Testing OpenCV Headless Mode:")
    
    try:
        # Set headless environment
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        
        import cv2
        import numpy as np
        
        # Create a test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "Test Image", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Test image operations
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Test saving image
        test_filename = "opencv_test.jpg"
        success = cv2.imwrite(test_filename, test_image)
        
        if success and os.path.exists(test_filename):
            print("   ✅ OpenCV headless mode working")
            print("   ✅ Image processing working")
            print("   ✅ Image saving working")
            
            # Clean up test file
            try:
                os.remove(test_filename)
            except:
                pass
            
            return True
        else:
            print("   ❌ Failed to save test image")
            return False
            
    except Exception as e:
        print(f"   ❌ OpenCV headless test failed: {e}")
        return False

def suggest_solutions(system_info, display_works, opencv_works):
    """Suggest solutions based on test results"""
    print("\n💡 Recommendations:")
    print("=" * 50)
    
    if system_info['ssh_detected']:
        print("🔗 SSH Connection Detected:")
        print("   ✅ Use headless mode: python bruno_headless.py")
        print("   ✅ Use web interface: python web_interface.py")
        print("   ✅ Use startup script: python start_bruno.py --mode headless")
    
    if not display_works:
        print("\n🖥️  Display Issues:")
        print("   ✅ Run in headless mode (no display needed)")
        print("   ✅ Use: python start_bruno.py --mode headless")
        if system_info['ssh_detected']:
            print("   ℹ️  SSH X11 forwarding: ssh -X username@hostname")
    
    if opencv_works:
        print("\n📹 OpenCV Solutions:")
        print("   ✅ OpenCV headless mode is working")
        print("   ✅ All image processing functions available")
        print("   ✅ Bruno can run without GUI")
    
    print("\n🚀 Quick Start Commands:")
    print("   Headless: python bruno_headless.py")
    print("   Web:      python web_interface.py")  
    print("   Auto:     python start_bruno.py")

def apply_fixes():
    """Apply common fixes"""
    print("\n🔧 Applying Common Fixes:")
    
    # Create .bashrc entries for environment variables
    bashrc_entries = [
        "# Bruno headless environment",
        "export QT_QPA_PLATFORM=offscreen",
        "export OPENCV_IO_ENABLE_OPENEXR=1"
    ]
    
    bashrc_path = os.path.expanduser("~/.bashrc")
    
    try:
        # Check if entries already exist
        if os.path.exists(bashrc_path):
            with open(bashrc_path, 'r') as f:
                content = f.read()
            
            if "QT_QPA_PLATFORM=offscreen" not in content:
                with open(bashrc_path, 'a') as f:
                    f.write("\n" + "\n".join(bashrc_entries) + "\n")
                print("   ✅ Added environment variables to ~/.bashrc")
            else:
                print("   ℹ️  Environment variables already in ~/.bashrc")
        
        print("   💡 Restart terminal or run: source ~/.bashrc")
        
    except Exception as e:
        print(f"   ❌ Failed to update ~/.bashrc: {e}")

def main():
    print("🤖 Bruno Qt/Display Issues Fixer")
    print("=" * 40)
    
    # Run diagnostic tests
    system_info = check_system_info()
    opencv_works = check_opencv_info()
    display_works = test_display_access()
    headless_works = test_opencv_headless()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY:")
    print("=" * 50)
    
    print(f"SSH Connection: {'✅ Yes' if system_info['ssh_detected'] else '❌ No'}")
    print(f"Display Access: {'✅ Yes' if display_works else '❌ No'}")
    print(f"OpenCV Available: {'✅ Yes' if opencv_works else '❌ No'}")
    print(f"Headless Mode: {'✅ Works' if headless_works else '❌ Failed'}")
    
    # Suggest solutions
    suggest_solutions(system_info, display_works, headless_works)
    
    # Apply fixes
    apply_fixes()
    
    print("\n" + "=" * 50)
    if headless_works:
        print("✅ READY: Bruno can run in headless mode!")
        print("🚀 Next: python start_bruno.py")
    else:
        print("❌ ISSUES: OpenCV headless mode not working")
        print("🔧 Try: pip install opencv-python-headless")

if __name__ == "__main__":
    main()