"""Tests for the bounded control layer in bruno_core.vla.

Everything here runs off-robot. The MuJoCo cases skip themselves when the
simulation extras are not installed, so the suite stays useful on Bruno.
"""

import math
import os
import sys
import time
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bruno_core.vla import kinematics as K
from bruno_core.vla.arm_state import ArmState, ArmTargetUnreachable
from bruno_core.vla.calibration import (
    CalibrationProfile,
    ServoCalibration,
    fit_servo_calibration,
)
from bruno_core.vla.contracts import Action, ContractError
from bruno_core.vla.controller import RobotController, RobotNotArmed
from bruno_core.vla.drivers.base import MockDriver


def action(**kwargs):
    return Action.from_dict({"confidence": 1.0, "reason": "test", **kwargs})


class KinematicsTest(unittest.TestCase):
    POSES = ((0, 6, 18, 0), (5, 13, 11, 0), (-5, 13, 11, 0), (0, 15, 6, -30), (3, 7, 22, 20))

    def test_ik_then_fk_returns_the_original_pose(self):
        for x, y, z, pitch in self.POSES:
            with self.subTest(pose=(x, y, z, pitch)):
                joints = K.ik_joint_degrees(x, y, z, pitch)
                self.assertIsNotNone(joints)
                (fx, fy, fz), got_pitch = K.forward_kinematics(joints)
                self.assertAlmostEqual(fx, x, places=3)
                self.assertAlmostEqual(fy, y, places=3)
                self.assertAlmostEqual(fz, z, places=3)
                self.assertAlmostEqual(got_pitch, pitch, places=3)

    def test_joint_and_servo_angles_round_trip(self):
        joints = K.ik_joint_degrees(0, 6, 18, 0)
        restored = K.servo_angles_to_joint_degrees(*K.joint_degrees_to_servo_angles(joints))
        for original, value in zip(joints, restored):
            self.assertAlmostEqual(original, value, places=9)

    def test_home_pose_puts_the_base_joint_at_zero(self):
        # Hiwonder measures forward as +y, so the home pose faces straight ahead.
        joints = K.ik_joint_degrees(0, 6, 18, 0)
        self.assertAlmostEqual(joints[0], 0.0, places=6)

    def test_unreachable_targets_return_none(self):
        self.assertIsNone(K.ik_joint_degrees(0, 60, 18, 0))

    def test_reachable_search_finds_a_pitch_inside_joint_travel(self):
        found = K.reachable_joint_degrees(0, 15, 6, 0)
        self.assertIsNotNone(found)
        _, joints = found
        for angle in joints:
            self.assertLessEqual(abs(angle), K.JOINT_LIMIT_DEG + 1e-6)


@unittest.skipUnless(
    os.path.isfile(os.path.join(REPO_ROOT, "third_party", "masterpi-mujoco", "masterpi.xml")),
    "MasterPi MJCF submodule is not checked out",
)
class MuJoCoAgreementTest(unittest.TestCase):
    """The whole sim-to-real story rests on these two agreeing."""

    @classmethod
    def setUpClass(cls):
        try:
            import mujoco
        except ImportError:
            raise unittest.SkipTest("mujoco is not installed")
        path = os.path.join(REPO_ROOT, "third_party", "masterpi-mujoco", "masterpi.xml")
        cls.mujoco = mujoco
        cls.model = mujoco.MjModel.from_xml_path(path)
        cls.data = mujoco.MjData(cls.model)
        cls.adr = {
            name: int(cls.model.joint(name).qposadr[0])
            for name in ("joint6_base_yaw", "joint5_shoulder", "joint4_elbow", "joint3_wrist")
        }

    def model_tcp_cm(self, joints_deg):
        self.data.qpos[:] = 0
        for name, angle in zip(self.adr, joints_deg):
            self.data.qpos[self.adr[name]] = math.radians(angle)
        self.mujoco.mj_forward(self.model, self.data)
        tcp = self.data.site("tcp").xpos - self.data.site("ik_origin").xpos
        # Model frame is +x forward; the Hiwonder frame is +y forward.
        return (-tcp[1] * 100.0, tcp[0] * 100.0, tcp[2] * 100.0)

    def test_our_kinematics_match_the_model_to_a_tenth_of_a_millimetre(self):
        for x, y, z, pitch in KinematicsTest.POSES:
            with self.subTest(pose=(x, y, z, pitch)):
                joints = K.ik_joint_degrees(x, y, z, pitch)
                mx, my, mz = self.model_tcp_cm(joints)
                error_mm = math.dist((mx, my, mz), (x, y, z)) * 10.0
                self.assertLess(error_mm, 0.1)


class ContractTest(unittest.TestCase):
    def test_joint_rotation_requires_a_known_joint(self):
        with self.assertRaises(ContractError):
            action(action="rotate_arm_by_deg", angle_deg=10)
        with self.assertRaises(ContractError):
            action(action="rotate_arm_by_deg", joint="knee", angle_deg=10)
        parsed = action(action="rotate_arm_by_deg", joint="base", angle_deg=10, duration_ms=400)
        self.assertEqual(parsed.joint, "base")

    def test_unknown_actions_and_out_of_range_values_are_rejected(self):
        for payload in (
            {"action": "fly", "speed": 0.5, "duration_ms": 100},
            {"action": "up", "speed": 5.0, "duration_ms": 100},
            {"action": "up", "speed": 0.5, "duration_ms": 99_000},
            {"action": "up", "speed": 0.0, "duration_ms": 100},
            {"action": "rotate_by_deg", "speed": 0.5, "angle_deg": 0.2, "duration_ms": 100},
        ):
            with self.subTest(payload=payload), self.assertRaises(ContractError):
                action(**payload)

    def test_stop_clears_every_motion_field(self):
        parsed = action(action="stop", speed=1.0, duration_ms=500, angle_deg=45, joint="base")
        self.assertEqual((parsed.speed, parsed.duration_ms, parsed.angle_deg, parsed.joint), (0.0, 0, 0.0, ""))

    def test_v2_payloads_are_refused(self):
        with self.assertRaises(ContractError):
            Action.from_dict({"schema": "bruno.vla.action.v2", "action": "stop"})


class CalibrationTest(unittest.TestCase):
    def test_profile_survives_a_dict_round_trip(self):
        profile = CalibrationProfile()
        self.assertEqual(CalibrationProfile.from_dict(profile.to_dict()).to_dict(), profile.to_dict())

    def test_pulse_conversion_round_trips_and_clamps(self):
        servo = ServoCalibration(channel=6)
        self.assertEqual(servo.pulse_for(0), 1500)
        self.assertAlmostEqual(servo.joint_deg_for(servo.pulse_for(45)), 45, places=1)
        # Beyond travel the joint angle clamps, so the pulse stays inside range.
        self.assertLessEqual(servo.pulse_for(400), servo.max_pulse)
        self.assertGreaterEqual(servo.pulse_for(-400), servo.min_pulse)

    def test_fit_recovers_a_known_linear_map(self):
        samples = [(q, 1500 + q * (2000 / 180)) for q in (-60, -30, 0, 30, 60)]
        fitted, worst = fit_servo_calibration(6, samples)
        self.assertTrue(fitted.verified)
        self.assertEqual(fitted.sign, 1)
        self.assertAlmostEqual(fitted.center_pulse, 1500, places=3)
        self.assertAlmostEqual(fitted.pulse_per_degree, 2000 / 180, places=5)
        self.assertLess(worst, 1e-6)

    def test_fit_detects_an_inverted_servo(self):
        samples = [(q, 1500 - q * 11.111) for q in (-60, 0, 60)]
        fitted, _ = fit_servo_calibration(5, samples)
        self.assertEqual(fitted.sign, -1)

    def test_a_nonlinear_relationship_is_not_marked_verified(self):
        samples = [(q, 1500 + q * 11.111 + (40 if q > 0 else -40)) for q in (-60, -30, 0, 30, 60)]
        fitted, worst = fit_servo_calibration(4, samples)
        self.assertFalse(fitted.verified)
        self.assertGreater(worst, 2.0)

    def test_degenerate_sample_sets_raise(self):
        with self.assertRaises(ValueError):
            fit_servo_calibration(6, [(0, 1500)])
        with self.assertRaises(ValueError):
            fit_servo_calibration(6, [(0, 1500), (0, 1600)])

    def test_rotation_duration_is_shared_by_both_backends(self):
        profile = CalibrationProfile()
        # 90 deg at half of 90 deg/s is two seconds, whichever driver runs it.
        self.assertEqual(profile.chassis.rotation_duration_ms(90, 0.5), 2000)

    def test_speed_maps_to_the_same_physical_rate_in_both_unit_systems(self):
        profile = CalibrationProfile()
        cmps = profile.chassis.linear_cmps(0.5)
        self.assertAlmostEqual(profile.sim_linear_mps(0.5), cmps / 100.0)
        self.assertAlmostEqual(profile.real_velocity_units(0.5), cmps * 10.0)


class ArmStateTest(unittest.TestCase):
    def setUp(self):
        self.state = ArmState(CalibrationProfile())

    def test_home_faces_forward_with_the_base_centred(self):
        self.assertAlmostEqual(self.state.joints["base"], 0.0, places=6)

    def test_cartesian_jog_moves_one_axis_by_the_step(self):
        before = list(self.state.position)
        self.state.jog_cartesian(action(action="move_arm_forward", speed=1.0, duration_ms=400))
        self.assertAlmostEqual(self.state.position[1], before[1] + self.state.profile.arm.step_cm)

    def test_jogging_stops_at_the_workspace_edge(self):
        for _ in range(40):
            try:
                self.state.jog_cartesian(action(action="move_arm_up", speed=1.0, duration_ms=400))
            except ArmTargetUnreachable:
                break
        self.assertLessEqual(self.state.position[2], self.state.profile.arm.max_z + 1e-6)

    def test_joint_rotation_applies_the_angle_and_updates_the_pose(self):
        self.state.rotate_joint("base", 30)
        self.assertAlmostEqual(self.state.joints["base"], 30.0, places=6)
        (x, y, _), _ = self.state.tcp_estimate()
        # +30 is counter-clockwise, so the tool swings to the robot's left (-x).
        self.assertLess(x, 0.0)
        self.assertAlmostEqual(math.hypot(x, y), 6.0, places=3)

    def test_a_single_step_is_capped(self):
        self.state.rotate_joint("base", 500)
        self.assertAlmostEqual(self.state.joints["base"], self.state.profile.arm.max_joint_step_deg)

    def test_a_joint_at_its_limit_refuses_to_go_further(self):
        for _ in range(10):
            try:
                self.state.rotate_joint("base", 30)
            except ArmTargetUnreachable:
                break
        self.assertAlmostEqual(self.state.joints["base"], K.JOINT_LIMIT_DEG)
        with self.assertRaises(ArmTargetUnreachable):
            self.state.rotate_joint("base", 30)


class DriverTest(unittest.TestCase):
    def test_identical_commands_produce_identical_joint_targets(self):
        left, right = MockDriver(), MockDriver()
        for step in (
            {"action": "move_arm_forward", "speed": 1.0, "duration_ms": 400},
            {"action": "rotate_arm_by_deg", "joint": "base", "angle_deg": 20, "duration_ms": 400},
            {"action": "move_arm_up", "speed": 0.5, "duration_ms": 400},
        ):
            command = action(**step)
            left.apply(command)
            right.apply(command)
        self.assertEqual(left.commanded_joints, right.commanded_joints)

    def test_translation_uses_the_calibrated_physical_speed(self):
        driver = MockDriver()
        driver.apply(action(action="right", speed=0.5, duration_ms=400))
        vx, vy, wz = driver.velocity
        self.assertAlmostEqual(vx, driver.profile.chassis.linear_cmps(0.5))
        self.assertAlmostEqual(vy, 0.0)
        self.assertAlmostEqual(wz, 0.0)

    def test_positive_rotation_is_counter_clockwise(self):
        driver = MockDriver()
        driver.apply(action(action="rotate_by_deg", angle_deg=45, speed=0.5, duration_ms=400))
        self.assertGreater(driver.velocity[2], 0.0)
        driver.apply(action(action="rotate_by_deg", angle_deg=-45, speed=0.5, duration_ms=400))
        self.assertLess(driver.velocity[2], 0.0)

    def test_hardware_refuses_joint_commands_without_a_verified_pulse_map(self):
        from bruno_core.vla.drivers.hardware import HardwareDriver, UncalibratedJoint

        driver = HardwareDriver(dry_run=True)
        # Cartesian moves go through the robot's own IK and need no calibration.
        driver.apply(action(action="move_arm_forward", speed=1.0, duration_ms=400))
        with self.assertRaises(UncalibratedJoint):
            driver.apply(action(action="rotate_arm_by_deg", joint="base", angle_deg=10, duration_ms=400))

    def test_hardware_allows_joint_commands_once_the_map_is_verified(self):
        from bruno_core.vla.drivers.hardware import HardwareDriver

        profile = CalibrationProfile(
            servos={
                name: ServoCalibration(channel=K.JOINT_SERVO_CHANNELS[name], verified=True)
                for name in K.JOINT_NAMES
            }
        )
        driver = HardwareDriver(profile=profile, dry_run=True)
        driver.apply(action(action="rotate_arm_by_deg", joint="base", angle_deg=10, duration_ms=400))
        self.assertAlmostEqual(driver.arm.joints["base"], 10.0)


class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.driver = MockDriver()
        self.controller = RobotController(self.driver, watchdog_seconds=0.4)
        self.addCleanup(self.controller.close)

    def test_starts_disarmed_and_refuses_motion(self):
        self.assertFalse(self.controller.status()["armed"])
        with self.assertRaises(RobotNotArmed):
            self.controller.execute(action(action="up", speed=0.3, duration_ms=200))

    def test_stop_is_accepted_while_disarmed(self):
        self.controller.execute(Action.stop("test"))

    def test_speed_and_duration_are_clamped(self):
        self.controller.set_armed(True)
        result = self.controller.execute(action(action="up", speed=1.0, duration_ms=5000))
        executed = result["executed"]
        self.assertLessEqual(executed["speed"], self.controller.max_speed)
        self.assertLessEqual(executed["duration_ms"], self.controller.max_duration_ms)

    def test_rotation_duration_comes_from_the_calibration(self):
        self.controller.set_armed(True)
        result = self.controller.execute(
            action(action="rotate_by_deg", angle_deg=90, speed=0.5, duration_ms=10)
        )
        expected = self.controller.profile.chassis.rotation_duration_ms(90, 0.5)
        self.assertEqual(result["executed"]["duration_ms"], expected)

    def test_low_confidence_becomes_a_stop(self):
        self.controller.set_armed(True)
        result = self.controller.execute(
            Action.from_dict({"action": "up", "speed": 0.3, "duration_ms": 200,
                              "confidence": 0.1, "reason": "unsure"})
        )
        self.assertEqual(result["executed"]["action"], "stop")

    def test_a_repeated_request_id_is_ignored(self):
        self.controller.set_armed(True)
        command = action(action="up", speed=0.3, duration_ms=200)
        self.controller.execute(command)
        before = self.driver.command_count
        self.assertTrue(self.controller.execute(command)["duplicate"])
        self.assertEqual(self.driver.command_count, before)

    def test_the_dead_man_timer_disarms_an_idle_robot(self):
        self.controller.set_armed(True)
        deadline = time.monotonic() + 3.0
        while self.controller.status()["armed"] and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(self.controller.status()["armed"])
        self.assertGreaterEqual(self.controller.status()["watchdog_trips"], 1)

    def test_a_driver_error_disarms_the_robot(self):
        class ExplodingDriver(MockDriver):
            def _drive(self, vx_cmps, vy_cmps, wz_deg_s):
                if vx_cmps or vy_cmps or wz_deg_s:
                    raise RuntimeError("motor bus fault")
                super()._drive(vx_cmps, vy_cmps, wz_deg_s)

        controller = RobotController(ExplodingDriver(), watchdog_seconds=5.0)
        self.addCleanup(controller.close)
        controller.set_armed(True)
        with self.assertRaises(RuntimeError):
            controller.execute(action(action="up", speed=0.3, duration_ms=200))
        self.assertFalse(controller.status()["armed"])
        self.assertIn("motor bus fault", controller.status()["last_driver_error"])


if __name__ == "__main__":
    unittest.main()


class ArmStateRollbackTest(unittest.TestCase):
    """A refused arm command must not advance our model of where the arm is."""

    def test_a_refused_joint_command_leaves_the_state_untouched(self):
        from bruno_core.vla.drivers.hardware import HardwareDriver, UncalibratedJoint

        driver = HardwareDriver(dry_run=True)
        before = dict(driver.arm.joints)
        with self.assertRaises(UncalibratedJoint):
            driver.apply(action(action="rotate_arm_by_deg", joint="base", angle_deg=20, duration_ms=400))
        self.assertEqual(driver.arm.joints, before)

    def test_a_refused_cartesian_move_leaves_the_state_untouched(self):
        class RefusingDriver(MockDriver):
            def _set_cartesian(self, position, pitch_deg, joints_deg, duration_ms):
                raise RuntimeError("IK rejected the target")

        driver = RefusingDriver()
        before = (list(driver.arm.position), dict(driver.arm.joints))
        with self.assertRaises(RuntimeError):
            driver.apply(action(action="move_arm_forward", speed=1.0, duration_ms=400))
        self.assertEqual(driver.arm.position, before[0])
        self.assertEqual(driver.arm.joints, before[1])


class DeviationTest(unittest.TestCase):
    """The robot's stored servo trim has to reach the wire."""

    def test_trim_shifts_the_pulse_without_changing_the_scale(self):
        plain = ServoCalibration(channel=6)
        trimmed = ServoCalibration(channel=6, deviation_us=-95.0)
        self.assertEqual(trimmed.pulse_for(0), plain.pulse_for(0) - 95)
        # A relative move is unaffected: only the absolute zero shifts.
        self.assertEqual(
            trimmed.pulse_for(20) - trimmed.pulse_for(0),
            plain.pulse_for(20) - plain.pulse_for(0),
        )

    def test_trim_round_trips(self):
        servo = ServoCalibration(channel=5, deviation_us=63.0)
        self.assertAlmostEqual(servo.joint_deg_for(servo.pulse_for(30)), 30, places=1)

    def test_trim_survives_a_config_round_trip(self):
        profile = CalibrationProfile(
            servos={
                name: ServoCalibration(channel=K.JOINT_SERVO_CHANNELS[name], deviation_us=-95.0)
                for name in K.JOINT_NAMES
            }
        )
        restored = CalibrationProfile.from_dict(profile.to_dict())
        self.assertEqual(restored.servos["base"].deviation_us, -95.0)


class IndividualWriteTest(unittest.TestCase):
    def test_each_channel_gets_its_own_board_call(self):
        """MasterPi's servosMove issues one write per channel; so must we."""
        from bruno_core.manipulation.arm import ArmConfig, ArmController

        calls = []

        class FakeBoard:
            def pwm_servo_set_position(self, move_time, pairs):
                calls.append((move_time, [list(p) for p in pairs]))

        controller = ArmController(cfg=ArmConfig(), board=FakeBoard(), arm_ik=object())
        controller.enabled = True
        controller.set_servos([(6, 1405), (5, 1563), (4, 1572), (3, 1559)], move_time=0.05)
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(len(pairs) == 1 for _, pairs in calls))
        self.assertEqual([pairs[0][0] for _, pairs in calls], [6, 5, 4, 3])


class ReachableTravelTest(unittest.TestCase):
    def test_trim_costs_travel_at_one_end(self):
        servo = ServoCalibration(channel=4, deviation_us=72.0)
        low, high = servo.reachable_deg()
        # +72 us of trim eats 72/11.111 = 6.5 deg off the top.
        self.assertAlmostEqual(high, 90.0 - 72.0 / (2000 / 180), places=1)
        self.assertAlmostEqual(low, -90.0, places=1)

    def test_no_trim_leaves_full_travel(self):
        low, high = ServoCalibration(channel=4).reachable_deg()
        self.assertAlmostEqual(low, -90.0, places=3)
        self.assertAlmostEqual(high, 90.0, places=3)


class ChassisRotationSignTest(unittest.TestCase):
    """Measured on the floor: a positive angle must turn the robot left."""

    def _rate_for(self, angle_deg):
        from bruno_core.vla.drivers.hardware import HardwareDriver

        sent = []

        class RecordingChassis:
            def set_velocity(self, speed, direction_deg, rotation):
                sent.append((speed, direction_deg, rotation))

            def stop(self):
                pass

        driver = HardwareDriver(dry_run=True)
        driver.chassis = RecordingChassis()
        driver.apply(action(action="rotate_by_deg", angle_deg=angle_deg,
                            speed=0.5, duration_ms=400))
        return sent[-1][2]

    def test_positive_angle_sends_a_positive_rate(self):
        # Hiwonder's angular_rate is counter-clockwise-positive, same as ours.
        self.assertGreater(self._rate_for(45), 0.0)

    def test_negative_angle_sends_a_negative_rate(self):
        self.assertLess(self._rate_for(-45), 0.0)


class StopHardeningTest(unittest.TestCase):
    """A stop that is silently dropped leaves the robot driving."""

    def _driver_with(self, chassis):
        from bruno_core.vla.drivers.hardware import HardwareDriver

        driver = HardwareDriver(dry_run=True)
        driver.chassis = chassis
        return driver

    def test_stop_is_sent_more_than_once(self):
        from bruno_core.vla.drivers import hardware

        class CountingChassis:
            def __init__(self):
                self.stops = 0

            def set_velocity(self, *args):
                pass

            def stop(self):
                self.stops += 1

        chassis = CountingChassis()
        self._driver_with(chassis).stop()
        self.assertEqual(chassis.stops, hardware.STOP_REPEAT_COUNT)

    def test_one_failed_attempt_does_not_raise_if_another_succeeds(self):
        class FlakyChassis:
            def __init__(self):
                self.calls = 0

            def set_velocity(self, *args):
                pass

            def stop(self):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("dropped write")

        self._driver_with(FlakyChassis()).stop()  # must not raise

    def test_total_failure_still_raises(self):
        class DeadChassis:
            def set_velocity(self, *args):
                pass

            def stop(self):
                raise RuntimeError("bus down")

        with self.assertRaises(RuntimeError):
            self._driver_with(DeadChassis()).stop()
