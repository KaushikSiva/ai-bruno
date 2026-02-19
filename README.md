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
- `bruno_core/`: shared camera, motion, sensors, safety, inference, audio, config, logging

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
