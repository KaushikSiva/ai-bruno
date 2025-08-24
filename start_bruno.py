#!/usr/bin/env python3
"""
Bruno Startup Script
Handles environment setup and runs Bruno in appropriate mode
"""

import os
import sys
import argparse

def setup_headless_environment():
    """Setup environment variables for headless operation"""
    # Prevent Qt display issues
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
    # Additional OpenCV environment variables
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    
    # Disable GUI backends
    os.environ['MPLBACKEND'] = 'Agg'  # For matplotlib if used
    
    print("✅ Headless environment configured")

def check_display_available():
    """Check if display is available"""
    display = os.environ.get('DISPLAY')
    
    if not display:
        print("ℹ️  No DISPLAY environment variable - running headless")
        return False
    
    try:
        # Try to connect to display
        import subprocess
        result = subprocess.run(['xset', 'q'], capture_output=True, timeout=5)
        if result.returncode == 0:
            print("✅ Display available")
            return True
    except:
        pass
    
    print("⚠️  Display not accessible - running headless")
    return False

def run_headless_mode(args):
    """Run Bruno in headless mode"""
    setup_headless_environment()
    
    print("🤖 Starting Bruno in HEADLESS mode")
    print("📱 Perfect for SSH connections and remote operation")
    
    # Import and run headless version
    try:
        from bruno_headless import main as headless_main
        
        # Set up arguments for headless mode
        sys.argv = ['bruno_headless.py']
        if args.config:
            sys.argv.extend(['--config', args.config])
        if args.log_level:
            sys.argv.extend(['--log-level', args.log_level])
        if args.no_save:
            sys.argv.append('--no-save')
        
        headless_main()
        
    except ImportError as e:
        print(f"❌ Failed to import headless mode: {e}")
        return False
    except Exception as e:
        print(f"❌ Error running headless mode: {e}")
        return False
    
    return True

def run_web_interface(args):
    """Run Bruno web interface"""
    setup_headless_environment()
    
    print("🌐 Starting Bruno Web Interface")
    print(f"🔗 Access at: http://localhost:{args.port}")
    
    try:
        from web_interface import main as web_main
        
        # Set up arguments for web interface
        sys.argv = ['web_interface.py']
        if args.host:
            sys.argv.extend(['--host', args.host])
        if args.port:
            sys.argv.extend(['--port', str(args.port)])
        if args.config:
            sys.argv.extend(['--config', args.config])
        
        web_main()
        
    except ImportError as e:
        print(f"❌ Failed to import web interface: {e}")
        return False
    except Exception as e:
        print(f"❌ Error running web interface: {e}")
        return False
    
    return True

def run_gui_mode(args):
    """Run Bruno with GUI (if display available)"""
    print("🖥️  Starting Bruno with GUI")
    
    try:
        from bruno_main import main as gui_main
        
        # Set up arguments
        sys.argv = ['bruno_main.py']
        if args.config:
            sys.argv.extend(['--config', args.config])
        if args.log_level:
            sys.argv.extend(['--log-level', args.log_level])
        if args.no_head:
            sys.argv.append('--no-head')
        if args.no_arm:
            sys.argv.append('--no-arm')
        
        gui_main()
        
    except ImportError as e:
        print(f"❌ Failed to import GUI mode: {e}")
        print("ℹ️  Falling back to headless mode...")
        return run_headless_mode(args)
    except Exception as e:
        if "qt.qpa.xcb" in str(e).lower() or "display" in str(e).lower():
            print("⚠️  Display issue detected - switching to headless mode")
            return run_headless_mode(args)
        else:
            print(f"❌ Error running GUI mode: {e}")
            return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Bruno - Bottle Detection Robot Launcher')
    parser.add_argument('--mode', '-m', 
                       choices=['auto', 'headless', 'web', 'gui'],
                       default='auto',
                       help='Run mode (auto=detect best mode)')
    parser.add_argument('--config', '-c', 
                       default='config/bruno_config.json',
                       help='Configuration file path')
    parser.add_argument('--log-level', '-l',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level')
    parser.add_argument('--host', default='0.0.0.0',
                       help='Web interface host (web mode only)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Web interface port (web mode only)')
    parser.add_argument('--no-head', action='store_true',
                       help='Disable head movement')
    parser.add_argument('--no-arm', action='store_true',
                       help='Disable arm movement')
    parser.add_argument('--no-save', action='store_true',
                       help='Disable saving detection images (headless mode)')
    
    args = parser.parse_args()
    
    print("🤖 Bruno - Bottle Detection Robot")
    print("=" * 40)
    
    # Determine run mode
    if args.mode == 'auto':
        display_available = check_display_available()
        ssh_connection = os.environ.get('SSH_CLIENT') or os.environ.get('SSH_CONNECTION')
        
        if ssh_connection:
            print("🔗 SSH connection detected - using headless mode")
            mode = 'headless'
        elif display_available:
            print("🖥️  Display available - using GUI mode")
            mode = 'gui'
        else:
            print("📱 No display - using headless mode")
            mode = 'headless'
    else:
        mode = args.mode
    
    print(f"🎯 Selected mode: {mode.upper()}")
    print("-" * 40)
    
    # Run in selected mode
    success = False
    
    if mode == 'headless':
        success = run_headless_mode(args)
    elif mode == 'web':
        success = run_web_interface(args)
    elif mode == 'gui':
        success = run_gui_mode(args)
    
    if success:
        print("✅ Bruno completed successfully")
    else:
        print("❌ Bruno failed to run")
        sys.exit(1)

if __name__ == "__main__":
    main()