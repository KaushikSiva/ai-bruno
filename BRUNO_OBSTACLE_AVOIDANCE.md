# Bruno Obstacle Avoidance System

This system provides intelligent obstacle detection and avoidance for your Bruno robot using computer vision and the existing movement control framework.

## Files Created

1. **`bruno_obstacle_avoidance.py`** - Main obstacle avoidance system (requires hardware)
2. **`bruno_avoidance_standalone.py`** - Standalone version for testing without hardware
3. **`simple_test_avoidance.py`** - Simple test script to verify functionality

## Key Features

### Obstacle Detection
- **Computer Vision Based**: Uses camera feed for obstacle detection
- **Edge Detection**: Detects obstacles using Canny edge detection
- **Distance Estimation**: Estimates obstacle distance based on size and position
- **Angle Calculation**: Determines obstacle position relative to robot center

### Avoidance Strategies
- **CONTINUE**: Safe to move forward (no obstacles detected)
- **TURN_LEFT/RIGHT**: Turn to avoid obstacles on one side
- **BACKUP_AND_TURN**: Back up and turn when obstacle is directly ahead
- **EMERGENCY_STOP**: Immediate stop when obstacle is too close

### Distance Zones
- **SAFE**: < 80px distance - Continue forward
- **CAUTION**: 80-120px distance - Execute avoidance maneuver  
- **DANGER**: > 120px distance - Emergency stop

## Configuration

The system is highly configurable:

```python
config = {
    'obstacle_threshold': 80,      # Distance to start avoiding (pixels)
    'danger_threshold': 120,       # Distance for emergency stop (pixels)
    'camera_url': 'http://127.0.0.1:8080?action=stream',
    'detection_area': {            # Area of frame to monitor
        'top': 0.3,               # 30% from top
        'bottom': 0.9,            # 90% from top  
        'left': 0.1,              # 10% from left
        'right': 0.9              # 90% from left
    },
    'avoidance_speed': 25,         # Speed when avoiding
    'normal_speed': 35,            # Normal forward speed
    'turn_time': 1.0,              # Time to turn
    'backup_time': 0.8,            # Time to backup
}
```

## Usage Instructions

### 1. Basic Usage (With Hardware)

```bash
python bruno_obstacle_avoidance.py
```

This runs the full system with:
- Real camera feed
- Bruno's movement controllers
- Real-time obstacle avoidance

### 2. Testing Without Hardware

```bash
python simple_test_avoidance.py
```

This runs automated tests with simulated obstacles to verify the detection logic.

### 3. Standalone Mode

```bash
python bruno_avoidance_standalone.py
```

This version works without hardware dependencies and can use your camera for testing.

## Integration with Original Script

The system adapts the original MasterPi obstacle avoidance concept to work with Bruno's architecture:

### Original Features Preserved:
- **Distance smoothing** using statistical filtering
- **Multiple obstacle handling**
- **Emergency stop functionality**
- **Visual feedback and debugging**

### Bruno-Specific Adaptations:
- **MovementController integration** for consistent motor control
- **RoombaNavigator compatibility** for autonomous navigation
- **Computer vision detection** instead of ultrasonic sensors
- **Configurable detection zones** for different environments

## Test Results

The test suite shows the system correctly:
- ✅ Detects obstacles in various positions
- ✅ Calculates appropriate distances and angles  
- ✅ Selects correct avoidance strategies
- ✅ Handles multiple obstacles intelligently
- ✅ Triggers emergency stops for close obstacles

Example test output:
```
Testing: Multiple obstacles
------------------------------
   Detected 2 obstacles
   Obstacle 1: Distance=106.0px, Angle=29.4 deg
   Obstacle 2: Distance=106.0px, Angle=-29.4 deg
   Action: TURN_LEFT
   Direction: LEFT
   Speed: 25%
   Emergency: False
   Message: AVOID: Obstacle to right, turning left
```

## Camera Setup

The system expects a camera stream at `http://127.0.0.1:8080?action=stream`. This is typically provided by:
- mjpg-streamer on Raspberry Pi
- USB camera streaming software
- IP camera with MJPEG stream

## Troubleshooting

### Camera Issues
- Verify camera stream is accessible
- Check URL format matches your setup
- Test with standalone version first

### Detection Issues
- Adjust `obstacle_threshold` and `danger_threshold` 
- Modify `detection_area` for your environment
- Check lighting conditions (edges need good contrast)

### Movement Issues  
- Verify Bruno's movement controllers are working
- Test individual movement commands separately
- Check motor connections and power

## Advanced Configuration

### Custom Detection Area
```python
'detection_area': {
    'top': 0.2,     # Monitor higher up
    'bottom': 0.95,  # Monitor lower down
    'left': 0.05,   # Wider monitoring
    'right': 0.95   # Wider monitoring
}
```

### Speed Profiles
```python
'normal_speed': 40,      # Faster normal speed
'avoidance_speed': 20,   # Slower avoidance
'turn_time': 1.5,        # Longer turns
'backup_time': 1.0,      # Longer backup
```

The system provides a robust foundation for autonomous navigation while maintaining Bruno's existing architecture and capabilities.