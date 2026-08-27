# Sim to real

Bruno's simulator and its hardware are not two implementations of the same
idea — they are one implementation with two output stages. This document
explains where the seam is, what is genuinely shared, and which numbers you
still have to measure on the floor.

## What is shared

```
                    Action (bruno.vla.action.v3)
                              |
                     RobotController          arming, bounds, dead-man
                              |
                        BaseDriver            what the command means
                              |
              +---------------+---------------+
              |                               |
        MuJoCoDriver                    HardwareDriver
     radians -> actuators            IK / pulses -> servos
```

Everything above the split is one code path: the action contract, the
workspace clamps, `ArmState`, the kinematics, and the calibration profile.
Given the same command, both drivers compute *the same joint vector*. That is
not an aspiration — `tests/test_vla.py` asserts it, and
`sim_real_check.py` re-checks it at runtime across a whole sequence.

Below the split they differ only in delivery:

| | simulation | hardware |
|---|---|---|
| chassis | body-frame velocity written to `drive_vx/vy/wz` | `mecanum.set_velocity(mm/s, heading, rad/s)` |
| arm, Cartesian | joint radians to position actuators | `ArmIK.setPitchRangeMoving` |
| arm, joint-space | joint radians to position actuators | PWM pulses on channels 6/5/4/3 |
| gripper | finger slide in metres | PWM pulse on channel 1 |

## The kinematics are verified, not assumed

`bruno_core/vla/kinematics.py` is a transcription of the IK the robot itself
runs (`ArmIK/InverseKinematics.py`), and `MuJoCoAgreementTest` checks our
solutions against the MJCF's own forward kinematics. Worst observed error
across the test poses is **0.016 mm**. The analytic FK in the same module
matches MuJoCo to floating-point noise.

This matters because it is what lets a joint angle be a shared currency. A
Cartesian pose alone would not be: it hides which elbow branch produced it.

## Rotation is positive counter-clockwise

`rotate_by_deg +45` turns the chassis left. `rotate_arm_by_deg base +45` swings
the arm left. This is the right-hand rule, matching the MuJoCo model and
REP-103.

Note that this is a deliberate break from the upstream `bruno_vla` project's
v2 schema, which defined chassis rotation as positive-clockwise while every
arm joint used the opposite convention.

Hiwonder's `set_velocity` turns out to take a counter-clockwise-positive
angular rate too, so the rate passes through unchanged. This was measured on the
floor rather than assumed: `HardwareDriver._drive` originally negated it, on the
belief that Hiwonder was clockwise-positive, and a commanded `+45` turned the
robot 45 degrees to the *right*.

Two things made that hard to catch earlier. With the wheels raised there is no
body rotation to observe -- only the wheels spinning, which look the same either
way unless you know which side is which. And `bruno_core/motion/mecanum.py`
supports the wrong belief: its `turn_left` passes a negative rate and
`turn_right` a positive one. Those labels appear to be swapped, but the rover
and face-follower apps may be built around them, so they are left as they are.

## What you must calibrate

Three things cannot be derived from a model and have to be measured.

### 1. The servo pulse map — required for joint commands

`rotate_arm_by_deg` writes PWM pulses directly. A wrong direction here does not
cause a small error, it drives the joint to the opposite limit, so the driver
**refuses joint-space commands until the map is verified**. Cartesian jogs are
unaffected: they go through the robot's own IK, which already knows the
conversion.

Run this on Bruno. Nothing moves — it reads the map out of
`ArmIK.transformAngelAdaptArm`, which is pure arithmetic:

```bash
python3 bruno_apps/teleop/calibrate_joints.py            # inspect the fit
python3 bruno_apps/teleop/calibrate_joints.py --write    # save it
```

A joint is only marked `verified` if a straight line actually fits its samples
to within 2 µs. If the fit is poor the relationship is not the linear one the
model assumes, and the numbers should not be trusted to drive a servo.

Defaults, pending measurement, come from the MJCF's own generator:
`500..2500 µs = 0..180°`, centre 1500, so 11.111 µs per degree.

### 2. Chassis speed — `vla.chassis.max_linear_speed_cmps`

Drive a measured distance at a known speed and time it:

```bash
python3 bruno_apps/teleop/main.py --target real up --speed 0.5 --duration-ms 2000
```

At speed 0.5 for 2 s the robot should cover `max_linear_speed_cmps` cm. Scale
the config value by whatever ratio you actually measure.

`real_velocity_units_per_cmps` (default 10.0) assumes Hiwonder's `set_velocity`
takes mm/s. If your MasterPi build disagrees, this is the single knob to fix
it, and it leaves the simulator untouched.

### 3. Rotation rate — `vla.chassis.max_rotation_deg_per_s`

`rotate_by_deg` is open loop: it spins for `angle / rate` seconds and stops.
Acceleration means the achieved angle always undershoots. In simulation the
shortfall is measurable, and `sim_real_check.py` prints it along with the
corrected value to use:

```
   commanded   spin time   achieved   shortfall
        +90d       2000ms     +84.4d       -5.6d
  Sim achieves 94% of the commanded angle (actuator ramp-up).
```

The real robot has the same effect with a different magnitude, and no way to
observe it. Measure it with a protractor and a floor mark before trusting
rotation angles on hardware.

## Running the check

```bash
python3 bruno_apps/teleop/sim_real_check.py                # sim vs the dry hardware path
python3 bruno_apps/teleop/sim_real_check.py --hardware live  # on Bruno, arm clear
python3 bruno_apps/teleop/sim_real_check.py --hardware none  # sim only
```

`--hardware dry` is useful off-robot: it runs the real IK bounds and pulse
arithmetic without a robot attached, so a pose the MasterPi would refuse shows
up before you are standing next to the hardware.

## What simulation cannot tell you

Bringing the arm up on hardware turned up a defect that no amount of simulation
would ever have caught, and it is worth understanding why.

The MasterPi stores a per-servo trim, `deviation_data`, which
`ArmIK.servosMove` adds to every pulse it writes. On this robot it is
`{3: +59, 4: +72, 5: +63, 6: -95}`. Calibration originally read
`transformAngelAdaptArm`, which returns the value *before* that trim, so every
pulse the driver wrote landed up to 8.5 degrees from where the robot's own IK
would have put it.

Sim and hardware agreed perfectly the whole time. Both computed the same joint
angles from the same command, and `sim_real_check.py` reported no disagreement,
because it compares commanded joints -- which were right. The error lived
entirely in the last step, turning a correct joint angle into a pulse, and only
the physical arm knew about it. It first showed up as an operator saying the
base looked about fifteen degrees off centre.

The lesson generalises: a shared contract makes sim predictive of *what will be
commanded*, never of *what the hardware will do with it*. Anything downstream of
the joint angle -- trim, backlash, a servo's true travel, gravity droop -- is
outside simulation's reach by construction. Test those on the robot, and treat
a clean sim run as evidence about your logic, not about your machine.

A second consequence: because the trim shifts a servo's whole pulse range, it
costs travel at one end. Hiwonder does not account for this -- its range check
runs on the pre-trim value, so at the extremes it will write past a servo's own
limit. This driver clamps instead, and `calibrate_joints.py` prints what travel
each joint actually has left.

## This chassis cannot strafe

`vla.chassis.supports_strafe` is **false** on this robot, and `left`/`right` are
refused with an explanation rather than executed.

Mecanum wheels only translate sideways when their rollers form an **X** viewed
from above: front-left and rear-right share one roller direction, front-right
and rear-left the other. On this robot they are mounted parallel, so the
sideways force components add into a torque instead of cancelling into
translation, and a strafe command spins the robot. Nothing in software can
correct that.

The symptom pattern is diagnostic, and worth recognising on any mecanum base:
forward, backward, and rotation all work perfectly, because none of them depend
on roller geometry. Strafing is the only motion that does, so it is the only one
that fails.

To fix it, re-mount the wheels so the top rollers form an X, then set
`supports_strafe` back to true.

The flag lives in the shared driver layer rather than the hardware driver, so
simulation refuses the command too. A sim that strafes a robot which cannot is
not predicting anything.

## Known limitations

**The MJCF base joints are world-frame.** The model gives the chassis
`base_x`/`base_y` slide joints *before* `base_yaw`, so their axes do not rotate
with the robot. `MuJoCoDriver._drive` compensates by rotating the command
through the current yaw, which restores body-frame behaviour. A command that
translated and rotated at once would still drift as the yaw changed underneath
it; the action contract issues one or the other, never both.

**Cartesian jogs only return the elbow-up branch.** After a joint rotation puts
the arm somewhere IK would not have chosen, the next Cartesian jog re-solves
from the pose and may reconfigure the arm. `ArmState` logs a warning when a jog
moves any joint by more than 45°.

**Put the arm at home before the first command.** Arm state is not persisted
between processes: every teleop invocation assumes the arm starts at the
configured home pose, and a joint command writes all four channels, not just
the one being rotated. So if the arm is physically somewhere else, the first
command moves it to home-plus-delta rather than nudging it from where it is --
over the command's duration, which may be fast. Either place the arm at home by
hand first, or make `home_arm` your first command with a generous duration:

```bash
python3 bruno_apps/teleop/main.py --target real home_arm --duration-ms 2000
```

Interactive `keys` mode keeps state for the length of the session, so this only
bites on the first command of each new process.

**Arm state is commanded, not measured.** The simulator can read true joint
angles back out of the model; the MasterPi's PWM servos have no feedback path
at all. `tcp_estimate_cm` on hardware is forward kinematics on what we asked
for, not where the arm is.

**Position targets outlive the dead-man timer.** The watchdog stops the
chassis and disarms, but a servo already told to travel to a position will
finish travelling. Keep arm durations short and keep a physical emergency stop
within reach.

**The arm frame differs from `arm_control`.** The `vla` config block uses
Hiwonder's convention — `+y` forward, home `(0, 6, 18)` — which is what the IK
and the MuJoCo model both expect. The older `arm_control` block used by
`pick_place` has `home_position: [15, 0, 20]`, i.e. `+x` forward. Under
Hiwonder IK that pose sits at a base yaw of −90°, pointing off the robot's
right side. The two blocks are independent, so nothing breaks, but they cannot
both describe the same physical home pose — worth resolving before the two apps
share arm calibration.
