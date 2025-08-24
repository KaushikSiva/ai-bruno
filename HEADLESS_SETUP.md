# Bruno Headless Setup - Fix Qt Display Issues

## The Problem
When running Bruno over SSH or on headless systems, you get this error:
```
qt.qpa.xcb: could not connect to display 
This application failed to start because no Qt platform plugin could be initialized.
```

## Quick Solutions

### 1. Run Headless Mode (Recommended)
```bash
python bruno_headless.py
```
- ✅ No display required
- ✅ Works over SSH  
- ✅ Saves detection images
- ✅ Full logging

### 2. Use Web Interface
```bash
python web_interface.py
```
Then visit: `http://your_robot_ip:5000`
- ✅ Monitor via browser
- ✅ Remote control
- ✅ Live camera feed

### 3. Auto-Detect Mode
```bash
python start_bruno.py
```
- ✅ Automatically chooses best mode
- ✅ Handles SSH connections
- ✅ Falls back gracefully

## Fix Environment Issues

### Step 1: Run Diagnostic
```bash
python fix_qt_issues.py
```
This will:
- Check your system setup
- Test OpenCV headless mode
- Apply common fixes
- Show recommendations

### Step 2: Set Environment Variables
```bash
export QT_QPA_PLATFORM=offscreen
export OPENCV_IO_ENABLE_OPENEXR=1
```

### Step 3: Make Permanent
Add to `~/.bashrc`:
```bash
echo "export QT_QPA_PLATFORM=offscreen" >> ~/.bashrc
source ~/.bashrc
```

## Comparison of Modes

| Mode | Display Needed | SSH Compatible | Web Access | GUI |
|------|----------------|----------------|------------|-----|
| **Headless** | ❌ No | ✅ Yes | ❌ No | ❌ No |
| **Web Interface** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **GUI** | ✅ Yes | ❌ No | ❌ No | ✅ Yes |

## Features Available in Headless Mode

✅ **Full Bottle Detection** - All computer vision features  
✅ **Movement Control** - Approaches bottles and stops at 1 foot  
✅ **Distance Estimation** - Accurate distance calculation  
✅ **Head Nodding** - Celebrates when bottles are reached  
✅ **Safety Systems** - Emergency stops and collision avoidance  
✅ **Status Logging** - Detailed progress information  
✅ **Image Saving** - Saves detection images for review  

## Troubleshooting

### Still Getting Qt Errors?
```bash
# Install headless OpenCV
pip install opencv-python-headless

# Remove regular OpenCV
pip uninstall opencv-python
```

### Web Interface Not Loading?
```bash
# Install Flask if missing
pip install flask

# Check firewall
sudo ufw allow 5000
```

### Camera Not Working?
```bash
# Test camera connection
python check_camera.py

# Run camera diagnostics  
python src/calibration/camera_test.py
```

## SSH Setup for Remote Access

### Enable SSH (if not already enabled)
```bash
sudo systemctl enable ssh
sudo systemctl start ssh
```

### Connect with X11 Forwarding (Optional)
```bash
ssh -X pi@your_robot_ip
```

### Use Web Interface for Best Experience
```bash
# On robot:
python web_interface.py --host 0.0.0.0

# On your computer:
# Open browser to: http://robot_ip:5000
```

## Quick Start Commands

```bash
# Diagnose issues
python fix_qt_issues.py

# Run headless (SSH-friendly)
python bruno_headless.py

# Run web interface (browser access)
python web_interface.py

# Auto-detect best mode
python start_bruno.py
```

## File Outputs

**Headless Mode Saves:**
- `detections/` - Detection images with timestamps
- `bruno_headless.log` - Detailed activity log

**Web Interface Provides:**
- Live camera stream
- Real-time status dashboard  
- Remote controls
- Activity logging

---

🎯 **Bottom Line:** Use `python bruno_headless.py` for SSH connections, or `python web_interface.py` for browser-based monitoring!