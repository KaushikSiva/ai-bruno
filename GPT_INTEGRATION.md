# Bruno GPT Vision Integration

This document describes how to use the enhanced GPT Vision integration with the Bruno robot project for intelligent bottle detection and automated pickup behavior.

## Overview

The `gpt.py` module provides AI-powered bottle detection using OpenAI's GPT Vision API, integrated with Bruno's existing robot control systems. It can detect both bottles and garbage bins, and automatically approach and interact with them.

## Features

- **AI-Powered Detection**: Uses OpenAI GPT Vision for intelligent object recognition
- **Fallback Detection**: Falls back to local OpenCV detection if API is unavailable
- **Integrated Control**: Uses Bruno's existing movement and head controllers
- **Configurable**: Works with Bruno's configuration system
- **Multiple Camera Support**: Supports both local cameras and network camera streams
- **Debug Mode**: Can save debug images and detection results

## Prerequisites

### Required Dependencies

```bash
# Core dependencies
pip install openai pillow numpy

# For local camera support
pip install opencv-python

# For network camera support
sudo apt-get install ffmpeg
```

### OpenAI API Key

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=sk-your-api-key-here
```

## Usage

### Basic Usage

```bash
# Run with default configuration
python3 gpt.py --config config/bruno_config.json

# Run in dry-run mode (no actual robot movement)
python3 gpt.py --config config/bruno_config.json --dry-run

# Use a specific camera URL
python3 gpt.py --camera-url http://127.0.0.1:8080?action=stream --dry-run
```

### Command Line Options

- `--config`: Path to Bruno configuration file (default: `config/bruno_config.json`)
- `--camera-url`: Override camera URL from config
- `--fps`: Camera FPS (default: 5)
- `--gpt-interval`: Seconds between API calls (default: 1.2, lower = more expensive)
- `--model`: OpenAI model for vision (default: `gpt-4o-mini`)
- `--dry-run`: Print actions instead of actuating robot
- `--save-debug`: Save debug images and detection results

### Configuration

The system uses Bruno's existing configuration file (`config/bruno_config.json`). Key sections:

```json
{
  "camera": {
    "device_id": "http://127.0.0.1:8080?action=stream",
    "width": 640,
    "height": 480,
    "fps": 30,
    "flip_horizontal": true
  },
  "movement_control": {
    "max_speed": 40,
    "min_speed": 15
  },
  "head_control": {
    "enabled": true
  }
}
```

## Behavior States

The system operates in the following states:

1. **SEARCH_BOTTLE**: Rotate and scan for bottles
2. **APPROACH_BOTTLE**: Move toward detected bottle
3. **GRAB_BOTTLE**: Simulate bottle pickup
4. **SEARCH_BIN**: Rotate and scan for garbage bins
5. **APPROACH_BIN**: Move toward detected bin
6. **DROP_BOTTLE**: Simulate dropping bottle in bin
7. **DONE**: Task complete

## Integration with Bruno Modules

### Robot Control

The system integrates with Bruno's existing control modules:

- **MovementController**: Handles robot chassis movement
- **HeadController**: Controls head movements and gestures
- **BottleDetector**: Provides fallback local detection

### Camera Support

Supports multiple camera types:

- **Local V4L2 cameras**: Uses OpenCV for capture
- **Network cameras**: Uses FFmpeg for MJPEG streams
- **Configurable parameters**: Resolution, FPS, horizontal flip

## Testing

Run the integration test to verify everything works:

```bash
python3 test_gpt_integration.py
```

This will test:
- Configuration loading
- Module imports
- Component creation
- Basic functionality

## Debug Mode

Enable debug mode to save images and detection results:

```bash
python3 gpt.py --save-debug --dry-run
```

This creates:
- `last_sent.jpg`: The image sent to GPT Vision API
- `last_result.json`: The detection results from the API

## Fallback Behavior

If the OpenAI API is unavailable or fails:

1. **Local Detection**: Falls back to Bruno's local OpenCV-based bottle detector
2. **No Bin Detection**: Local detector only finds bottles, not bins
3. **Graceful Degradation**: System continues to operate with reduced functionality

## Cost Considerations

- **API Calls**: Each detection requires an API call to OpenAI
- **Image Quality**: Lower quality images reduce bandwidth and cost
- **Detection Interval**: Longer intervals reduce API usage
- **Model Choice**: `gpt-4o-mini` is cheaper than `gpt-4o`

## Troubleshooting

### Common Issues

1. **OpenAI API Key Not Set**
   ```
   Warning: OPENAI_API_KEY not set. Will use local detection only.
   ```
   Solution: Set the environment variable

2. **Camera Not Found**
   ```
   Failed to start camera
   ```
   Solution: Check camera device ID or network URL

3. **Bruno Modules Not Available**
   ```
   Warning: Bruno modules not available
   ```
   Solution: Ensure all Bruno modules are properly installed

4. **FFmpeg Not Found**
   ```
   FFmpeg not found. Install with: sudo apt-get install -y ffmpeg
   ```
   Solution: Install FFmpeg for network camera support

### Performance Tuning

- **Detection Interval**: Increase `--gpt-interval` to reduce API costs
- **Image Quality**: Lower JPEG quality in `encode_jpeg_from_bgr()` to reduce bandwidth
- **Camera FPS**: Lower FPS reduces processing load
- **Movement Speed**: Adjust speed parameters in configuration

## Example Workflow

1. **Setup**: Install dependencies and set API key
2. **Test**: Run integration test to verify setup
3. **Dry Run**: Test with `--dry-run` to see behavior
4. **Live Run**: Remove `--dry-run` for actual robot operation
5. **Monitor**: Watch for detection results and robot behavior
6. **Debug**: Use `--save-debug` to analyze detection performance

## Future Enhancements

- **Arm Control Integration**: Add actual bottle pickup and drop functionality
- **Multi-Object Detection**: Detect and handle multiple bottles
- **Path Planning**: Implement more sophisticated navigation
- **Learning**: Add ability to learn from successful pickups
- **Web Interface**: Integrate with Bruno's web interface
