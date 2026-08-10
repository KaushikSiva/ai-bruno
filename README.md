# Bruno

YOUTUBE

[![Demo Video](https://img.youtube.com/vi/d_K3juzPHkk/hqdefault.jpg)](https://youtu.be/d_K3juzPHkk)
 

## Apps

This repo has four runnable apps plus shared core modules.

## Layout

- `bruno_apps/rover/`: navigation + camera safety + VLM advisory
- `bruno_apps/surveillance/`: snapshot/caption/summary + ultrasonic safety
- `bruno_apps/face_follower/`: face detection + pan/tilt tracking
- `bruno_apps/buddy/`: wake-word voice buddy
- `bruno_apps/pick_place/`: prompt-driven pick and place with the arm
- `bruno_core/`: shared camera, motion, sensors, safety, inference, audio, config, logging, manipulation

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Run

### Rover

Run in three terminals:

```bash
./bruno_apps/rover/run/start_camera.sh
./bruno_apps/rover/run/start_lfm.sh
./bruno_apps/rover/run/start_rover.sh
```

Defaults:
- Camera URL: `http://127.0.0.1:8080/`
- Local VLM base: `http://127.0.0.1:8081/v1`
- Rover mode: `builtin`

### Surveillance

```bash
./bruno_apps/surveillance/run/start_surveillance.sh
```

Default mode is `builtin`.

### Face Follower

```bash
./bruno_apps/face_follower/run/start_face_follower.sh
```

Defaults:
- Mode: `external`
- Scan speed: `1.5`
- Face confidence threshold: `0.6`
- Headless flag defaults to `--headless` unless `HEADLESS_FLAG` is explicitly set empty

Useful overrides:

```bash
MODE=builtin ./bruno_apps/face_follower/run/start_face_follower.sh
HEADLESS_FLAG="" ./bruno_apps/face_follower/run/start_face_follower.sh --debug
```

### Pick and Place

Calibrate the arm once (writes poses into `config/bruno_config.json`):

```bash
python3 bruno_apps/pick_place/calibrate_arm.py
```

Then run with the camera and VLM servers up (same two terminals as the rover):

```bash
./bruno_apps/pick_place/run/start_pick_place.sh "pick up the blue bottle and put it on the left"
```

The VLM is asked once, up front, which object the prompt refers to and where to
put it. Grasp geometry comes from CV (HSV blob + focal-length distance), not the
model — single-frame depth is not reliable enough to drive the arm.

Defaults:
- Grasp standoff: `18` cm (`PP_STANDOFF_CM`)
- Approach speed: `25` (`PP_SPEED`)
- Trackable colors: `clear`, `blue`, `green` (from `detection.color_detection`)

Dry run (perception and planning, no motion):

```bash
python3 bruno_apps/pick_place/main.py --prompt "pick up the green cup" --dry-run
```

### Buddy

```bash
./bruno_apps/buddy/run/start_buddy.sh
```

Defaults:
- Wake phrase from `BUDDY_WAKE` (current default: `hello`)
- Microphone index from `BUDDY_MIC_INDEX` (current default: `0`)
- Voice: `Dominoux`
- TTS off unless `AUDIO_FLAG=--audio` is provided

Example:

```bash
AUDIO_FLAG=--audio ./bruno_apps/buddy/run/start_buddy.sh
```

## Important Notes

- Do not run two processes that open the same `/dev/video*` camera at the same time.
- `builtin` mode reads from `BRUNO_CAMERA_URL` (HTTP stream), while `external` mode opens `/dev/video*` directly.
- For OpenAI-compatible LLM servers, `LLM_API_BASE` should usually include `/v1` (for example: `http://10.0.0.73:1234/v1`).
- Shared logs are under `logs/`.
