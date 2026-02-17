# Bruno Apps

Repo is now split into three independent runtimes plus shared core modules.

## Layout

- `bruno_apps/rover/`: camera navigation + camera safety + VLM advisory
- `bruno_apps/surveillance/`: snapshot/caption/summary + ultrasonic safety
- `bruno_apps/buddy/`: wake-word voice buddy
- `bruno_core/`: shared camera, motion, sensors, safety, inference, audio, config, logging

## Run

### Rover

```bash
./bruno_apps/rover/run/start_camera.sh
./bruno_apps/rover/run/start_lfm.sh
./bruno_apps/rover/run/start_rover.sh
```

### Surveillance

```bash
./bruno_apps/surveillance/run/start_surveillance.sh
```

### Buddy

```bash
./bruno_apps/buddy/run/start_buddy.sh
```

## Notes

- All launch scripts use system `python3` (no venv required).
- Rover local VLM defaults to `http://127.0.0.1:8081/v1`.
- MJPEG live stream defaults to `http://0.0.0.0:8080/`.
- Shared logs are under `logs/`.

