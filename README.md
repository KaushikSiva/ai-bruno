# Bruno

YOUTUBE

[![Demo Video](https://img.youtube.com/vi/d_K3juzPHkk/hqdefault.jpg)](https://youtu.be/d_K3juzPHkk)
 

## Apps

This repo has five runnable apps plus shared core modules.

## Layout

- `bruno_apps/rover/`: navigation + camera safety + VLM advisory
- `bruno_apps/surveillance/`: snapshot/caption/summary + ultrasonic safety
- `bruno_apps/face_follower/`: face detection + pan/tilt tracking
- `bruno_apps/buddy/`: wake-word voice buddy
- `bruno_apps/pick_place/`: prompt-driven pick and place with the arm
- `bruno_apps/teleop/`: hand control of the chassis and arm, in MuJoCo or on the robot
- `bruno_core/`: shared camera, motion, sensors, safety, inference, audio, config, logging, manipulation
- `bruno_core/vla/`: bounded action contract, safety controller, and the sim/hardware drivers
- `third_party/masterpi-mujoco/`: pinned MuJoCo model of the MasterPi (git submodule)

## Setup

```bash
python3 -m pip install -r requirements.txt
```

For the simulator, also fetch the robot model and its extra dependencies. These
are not needed on Bruno itself:

```bash
git submodule update --init --recursive
python3 -m pip install -r requirements-sim.txt
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

### Teleop

Drive the chassis and arm by hand. The same commands reach MuJoCo or the real
robot, because both go through `bruno_core/vla` and share one calibration
profile — see [the sim-to-real notes](bruno_apps/teleop/docs/SIM_TO_REAL.md).

Interactive, in simulation:

```bash
./bruno_apps/teleop/run/start_teleop.sh
```

`w/s/a/d` or the arrow keys drive, `q/e` turn by the angle step, `i/k j/l u/o`
jog the arm, `1`–`8` rotate the four arm joints by degrees, `[`/`]` work the
gripper, `h` homes the arm, space stops, `?` shows the keys.

One-shot commands take the same flags. Rotation is positive counter-clockwise,
so `+45` turns left:

```bash
python3 bruno_apps/teleop/main.py --target sim rotate_by_deg --angle 45
python3 bruno_apps/teleop/main.py --target sim rotate_arm_by_deg --joint base --angle 20
python3 bruno_apps/teleop/main.py --target sim status
```

To watch it in the MuJoCo viewer, run the bridge in one terminal and point
teleop at it in another:

```bash
./bruno_apps/teleop/run/start_sim.sh
TARGET=sim ROBOT_URL=http://127.0.0.1:8091 ./bruno_apps/teleop/run/start_teleop.sh
```

On the robot, calibrate the arm servo pulse map once — joint-space commands are
refused until it is verified, because a wrong direction drives a joint to the
opposite limit:

```bash
python3 bruno_apps/teleop/calibrate_joints.py --write
TARGET=real ./bruno_apps/teleop/run/start_teleop.sh
```

Check that simulation and hardware still agree, and see what remains to be
measured on the floor:

```bash
python3 bruno_apps/teleop/sim_real_check.py
```

Defaults:
- Target: `sim` (`TARGET=real` for the robot)
- Speed: `0.3`, burst `400` ms, turn step `15` deg, joint step `10` deg
- Bridge: `http://127.0.0.1:8091`, starts disarmed

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
- Teleop and the VLA bridge start **disarmed** and disarm again if commands stop arriving. Arm and gripper commands are position targets, so they finish travelling even after a watchdog stop — keep a physical emergency stop within reach.
- `BRUNO_SHARED_TOKEN` is required whenever the bridge binds to anything other than loopback.
