# Bruno Face Follow System

A comprehensive face detection and tracking system for the MasterPi robotics platform, implementing systematic scanning and servo-based face following.

## Features

### 🔍 **Systematic Face Scanning**
- **Arm Movement Pattern**: Based on HiWonder section 4.4.3 kinematics
- **Scanning Coordinates**: (0,6,18) → (5,6,18) → (5,13,11) → (0,13,11) → (-5,13,11) → (-5,6,18) → center
- **Configurable Speed**: Adjustable scanning speed via command line
- **Continuous Scanning**: Loops until face is detected

### 🎯 **Face Detection & Recognition**  
- **MediaPipe Integration**: Based on HiWonder section 5.7 implementation
- **High Confidence Detection**: 0.8 minimum confidence threshold
- **Multi-face Handling**: Tracks largest/closest face when multiple detected
- **Robust Recognition**: Handles various lighting conditions and angles

### 📷 **Camera Servo Tracking**
- **Smooth Following**: PID-like servo control for steady tracking
- **Horizontal Panning**: Servo 6 control with 1100-1900 pulse range
- **Dead Zone Control**: 30-pixel center tolerance to reduce jitter
- **Dynamic Adjustment**: Real-time face centering

### 🤖 **State Machine Architecture**
```
INITIALIZING → SCANNING → FACE_DETECTED → TRACKING → FACE_LOST → SCANNING
                    ↓
              GREETING (audio + head nod)
```

### 🔊 **Audio & Visual Feedback**
- **Greeting System**: Nods head and speaks when face first detected
- **Multiple Greetings**: Random selection from greeting pool
- **Visual Debug**: Real-time state and detection info overlay
- **Distance Feedback**: Face area-based distance categorization

## Files

### Core Implementation
- **`face_follow_test.py`**: Main face following system
- **`test_face_follow_components.py`**: Component testing without hardware dependencies

### Key Classes
- **`FaceFollowTest`**: Main controller integrating all subsystems
- **`ArmScanner`**: Handles systematic arm movements for face search
- **`FaceTracker`**: MediaPipe-based face detection and tracking
- **`CameraMountController`**: Servo-based camera positioning
- **`FaceFollowState`**: State machine enumeration

## Usage

### Basic Usage
```bash
# Basic face following
python3 face_follow_test.py --mode external

# With audio greetings and debug info  
python3 face_follow_test.py --mode external --audio --voice Dominus --debug

# Adjust scanning speed
python3 face_follow_test.py --mode external --scan-speed 2.0 --debug
```

### Command Line Options
- `--mode`: Camera mode (`builtin` or `external`)
- `--audio`: Enable audio greetings  
- `--voice`: TTS voice selection (default: Dominus)
- `--scan-speed`: Scanning speed in seconds (default: 1.5)
- `--debug`: Enable debug information overlay

### Keyboard Controls
- **ESC**: Exit program
- **R**: Manual reset (restart scanning)

## Technical Implementation

### Hardware Requirements
- **MasterPi Robot Platform**: With arm kinematics support
- **Camera**: Builtin or external USB camera
- **Servos**: 
  - Servo 3: Head movement (greeting nod)
  - Servo 6: Camera horizontal panning
- **Audio**: Speaker for TTS greetings (optional)

### Dependencies
```bash
pip install opencv-python mediapipe numpy
```

### State Machine Logic

#### SCANNING State
- Execute systematic arm movement pattern
- Continuously check for faces in camera feed
- Transition to FACE_DETECTED when face found

#### FACE_DETECTED State
- Verify face detection consistency
- Center camera servo on detected face
- Execute greeting sequence (nod + audio)
- Transition to TRACKING state

#### TRACKING State  
- Continuously track face with camera servo
- Monitor face distance and provide feedback
- Handle temporary face loss (grace period)
- Transition to FACE_LOST if face disappears too long

#### FACE_LOST State
- Quick re-scan for face in current view
- Return to SCANNING if face not immediately found
- Reset greeting flag for next detection cycle

### Performance Characteristics
- **Face Detection**: ~30 FPS on Raspberry Pi 4
- **Servo Response**: ~50ms camera tracking updates
- **Scanning Cycle**: ~10-15 seconds full sweep (configurable)
- **Grace Period**: 2 seconds before declaring face lost

## Integration with Bruno System

### Audio Integration
- Uses existing `audio_tts.py` TTS system
- Supports multiple voice options (Dominus, Ashley, etc.)
- Automatic greeting on first face detection

### Camera Integration  
- Leverages `camera_shared.py` infrastructure
- Support for both builtin and external cameras
- Automatic reconnection handling

### Hardware Integration
- Uses MasterPi SDK (`ros_robot_controller_sdk`)
- Kinematics integration (`arm_move_ik.ArmIK`)
- Safe fallback when hardware unavailable

## Testing

### Component Testing
```bash
# Test all components without hardware
python3 test_face_follow_components.py
```

Test results show:
- ✅ Scanning pattern execution
- ✅ Camera tracking logic
- ✅ State machine transitions  
- ✅ Integrated behavior simulation

### Expected Behavior
1. **Startup**: Robot begins systematic arm scanning
2. **Detection**: Stops scanning when face found, centers camera
3. **Greeting**: Nods head and speaks greeting message
4. **Tracking**: Smoothly follows face with camera servo
5. **Recovery**: Returns to scanning if face is lost

## Troubleshooting

### Common Issues
- **No face detection**: Check lighting, camera focus, detection confidence
- **Jerky tracking**: Adjust dead zone, servo speed, or detection frequency  
- **Scanning too fast/slow**: Use `--scan-speed` parameter
- **Audio issues**: Check TTS configuration, speaker connection
- **Hardware errors**: Verify MasterPi SDK installation and servo connections

### Debug Mode
Enable `--debug` flag to see:
- Real-time state information
- Face detection confidence scores
- Servo tracking adjustments
- Performance metrics (FPS)

## Future Enhancements

- **3D Face Tracking**: Add vertical servo control for full pan/tilt
- **Face Recognition**: Store and recognize specific individuals  
- **Multi-person Tracking**: Handle multiple faces intelligently
- **Distance Control**: Move robot base to maintain optimal face distance
- **Emotion Detection**: Respond differently based on facial expressions

---

**Note**: This system requires the MasterPi hardware platform and associated SDK. For development and testing without hardware, use the included component test script.