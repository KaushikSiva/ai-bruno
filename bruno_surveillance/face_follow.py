#!/usr/bin/env python3
# coding: utf-8
"""
Bruno Face Follow (standalone)

Behavior
- Opens camera (builtin/external via --mode)
- Detects a face via MediaPipe FaceDetection and tracks its center
- Greets once (first confirmed face) with nod + optional "Hello"
- Follows by centering on the face and maintaining distance
- Uses ultrasonic to avoid getting too close

Run
  python3 bruno_surveillance/face_follow.py --mode external --audio --voice Ashley

Dependencies
- mediapipe, opencv-python
- MasterPi SDK for motion & ultrasonic
- bruno_surveillance/audio_tts.py for TTS (optional)
"""
import os
import sys
import time
import argparse
from typing import Optional

sys.path.append('/home/pi/MasterPi')

import cv2
import mediapipe as mp

from utils import LOG, paths
from camera_shared import make_camera, read_or_reconnect
from robot_motion import MecanumWrapper
from ultrasonic import UltrasonicRGB
from audio_tts import TTSSpeaker

try:
    from common.ros_robot_controller_sdk import Board
    HW_BOARD = True
except Exception:
    Board = None
    HW_BOARD = False


class FaceFollower:
    def __init__(self, mode: str, audio: bool, voice: str):
        self.camera = make_camera(mode, retry_attempts=3, retry_delay=2.0)
        self.motion = MecanumWrapper(forward_speed=40, turn_speed=40)
        self.ultra = UltrasonicRGB()
        self.speaker = TTSSpeaker(enabled=audio, voice=voice) if audio else None
        if self.speaker:
            self.speaker.start()

        self.board = Board() if HW_BOARD else None
        # Greeting + tracking state
        self.greeted_once = False
        self.last_bbox: Optional[tuple] = None  # (x, y, w, h) for last detected face
        self.last_detect_ts: float = 0.0

        # Smoothing (EMA) for center and area
        self.cx_smooth: Optional[float] = None
        self.area_smooth: Optional[float] = None
        self.ema_alpha = 0.4

        # Face detector (MediaPipe)
        self.min_detection_conf = 0.6
        try:
            # Prefer model_selection=1 for farther range; fall back if older mediapipe
            self.face_detection = mp.solutions.face_detection.FaceDetection(
                min_detection_confidence=self.min_detection_conf, model_selection=1
            )
        except TypeError:
            self.face_detection = mp.solutions.face_detection.FaceDetection(
                min_detection_confidence=self.min_detection_conf
            )

        # Follow tuning
        # If your chassis turns the opposite way, set invert_yaw = False/True
        self.invert_yaw = True  # many chassis map yaw inverted; flip if needed
        self.center_dead_px = 30  # face needs tighter centering than legs
        # Use relative area thresholds (ratio of face box to frame area)
        # These values work across common resolutions (e.g., 640x480)
        self.area_ratio_far = 0.010   # <1.0% of frame: move forward
        self.area_ratio_target = 0.030  # ~3% of frame: good distance target
        self.area_ratio_near = 0.060  # >6% of frame: too close, stop

        # Timing
        self.loop_sleep = 0.03
        self.keep_bbox_for = 0.7  # seconds to keep last box if temporarily lost
        self.debug = False

    def _nod(self):
        if not self.board:
            return
        try:
            # simple nod on servo 3
            for v, t in ((500, 0.15), (900, 0.15), (500, 0.15), (700, 0.15)):
                self.board.pwm_servo_set_position(0.1, [[3, v]])
                time.sleep(t)
        except Exception:
            pass

    def _greet(self):
        if self.greeted_once:
            return
        self.greeted_once = True
        try:
            self._nod()
            if self.speaker:
                self.speaker.speak_sync("Hello")
        except Exception:
            pass

    def _follow_step(self, frame_w, frame_h, bbox):
        # Safety: ultrasonic
        d = self.ultra.get_distance_cm()
        if d is not None and d <= 25:
            self.motion.stop(); return

        (x, y, w, h) = bbox
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        frame_area = max(1, frame_w * frame_h)
        area_ratio = area / frame_area

        # Smooth center x and area to reduce jitter
        if self.cx_smooth is None:
            self.cx_smooth = float(cx)
        else:
            self.cx_smooth = (1 - self.ema_alpha) * self.cx_smooth + self.ema_alpha * float(cx)
        if self.area_smooth is None:
            self.area_smooth = float(area)
        else:
            self.area_smooth = (1 - self.ema_alpha) * self.area_smooth + self.ema_alpha * float(area)
        cx_eff = int(self.cx_smooth)
        area_ratio_eff = float(self.area_smooth) / frame_area

        # Heading control (discrete)
        dx = cx_eff - (frame_w // 2)
        turned = False
        if dx < -self.center_dead_px:
            # target is to the left of center
            if self.invert_yaw:
                self.motion.turn_right(0.12)
            else:
                self.motion.turn_left(0.12)
            turned = True
        elif dx > self.center_dead_px:
            # target is to the right of center
            if self.invert_yaw:
                self.motion.turn_left(0.12)
            else:
                self.motion.turn_right(0.12)
            turned = True

        # If we just executed a turn, skip forward motion this cycle to avoid stacked blocking calls
        if turned:
            return

        # Range control (based on relative face size)
        if area_ratio_eff < self.area_ratio_far:
            self.motion.forward(0.2)
        elif area_ratio_eff > self.area_ratio_near:
            self.motion.stop()
        else:
            # in band: gentle forward if still below target
            if area_ratio_eff < self.area_ratio_target:
                self.motion.forward(0.1)
            else:
                self.motion.stop()

    def _detect_face_bbox(self, frame_bgr) -> tuple | None:
        """Detect the most confident face and return pixel bbox (x,y,w,h) or None."""
        try:
            h, w = frame_bgr.shape[:2]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self.face_detection.process(rgb)
            if not res.detections:
                return None
            best = None
            best_area = 0
            for det in res.detections:
                score = det.score[0] if det.score else 0.0
                if score < self.min_detection_conf:
                    continue
                rb = det.location_data.relative_bounding_box
                x = max(0, int(rb.xmin * w))
                y = max(0, int(rb.ymin * h))
                ww = max(0, int(rb.width * w))
                hh = max(0, int(rb.height * h))
                if ww <= 0 or hh <= 0:
                    continue
                area = ww * hh
                if area > best_area:
                    best = (x, y, ww, hh)
                    best_area = area
            return best
        except Exception:
            return None

    def _draw_debug(self, frame_bgr, bbox: Optional[tuple]):
        if not self.debug or frame_bgr is None:
            return
        try:
            h, w = frame_bgr.shape[:2]
            # center lines
            cv2.line(frame_bgr, (w//2, 0), (w//2, h), (0, 255, 255), 1)
            cv2.line(frame_bgr, (0, h//2), (w, h//2), (0, 255, 255), 1)
            if bbox is not None:
                x, y, bw, bh = bbox
                cv2.rectangle(frame_bgr, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
                cx = x + bw//2
                cy = y + bh//2
                cv2.circle(frame_bgr, (cx, cy), 4, (0, 255, 0), -1)
                area = bw * bh
                ratio = area / max(1, w*h)
                cv2.putText(frame_bgr, f"ratio={ratio:.3f}", (x, max(0, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imshow('FaceFollow Debug', cv2.resize(frame_bgr, (640, 480)))
            cv2.waitKey(1)
        except Exception:
            pass

    def run(self):
        LOG.info('🙂 Face Follow (MediaPipe): greet + nod + follow')
        if not self.camera.open():
            LOG.error('❌ Cannot open camera')
            return
        last_frame = None
        try:
            while True:
                last_frame = read_or_reconnect(self.camera, last_frame)
                if last_frame is None:
                    time.sleep(self.loop_sleep); continue
                h, w = last_frame.shape[:2]
                # Detect face every loop; keep last bbox briefly when transiently lost
                box = self._detect_face_bbox(last_frame)
                now = time.time()
                if box is not None and box[2] > 0 and box[3] > 0:
                    self.last_bbox = box
                    self.last_detect_ts = now
                    if not self.greeted_once:
                        self._greet()
                # Draw debug overlay if requested
                self._draw_debug(last_frame, self.last_bbox)

                if self.last_bbox is not None and (now - self.last_detect_ts) <= self.keep_bbox_for:
                    self._follow_step(w, h, self.last_bbox)
                else:
                    self.motion.stop()

                time.sleep(self.loop_sleep)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.motion.stop()
            except Exception:
                pass


def main():
    p = argparse.ArgumentParser(description='Bruno Face Follow (standalone)')
    p.add_argument('--mode', choices=['builtin','external'], default=os.environ.get('CAM_MODE','external'))
    p.add_argument('--audio', action='store_true', help='Enable greeting voice')
    p.add_argument('--voice', default=os.environ.get('BRUNO_AUDIO_VOICE','Ashley'))
    p.add_argument('--invert-yaw', type=int, choices=[0,1], default=None, help='Override yaw inversion (1=invert,0=normal)')
    p.add_argument('--dead-px', type=int, default=None, help='Center deadzone in pixels')
    p.add_argument('--min-conf', type=float, default=None, help='Min face detection confidence (0-1)')
    p.add_argument('--far-ratio', type=float, default=None, help='Face area ratio below which to move forward')
    p.add_argument('--target-ratio', type=float, default=None, help='Target face area ratio')
    p.add_argument('--near-ratio', type=float, default=None, help='Face area ratio above which to stop')
    p.add_argument('--debug', action='store_true', help='Show debug overlay window')
    args = p.parse_args()

    app = FaceFollower(args.mode, args.audio, args.voice)
    if args.invert_yaw is not None:
        app.invert_yaw = bool(args.invert_yaw)
    if args.dead_px is not None:
        app.center_dead_px = max(5, args.dead_px)
    if args.min_conf is not None:
        app.min_detection_conf = max(0.1, min(0.9, args.min_conf))
    if args.far_ratio is not None:
        app.area_ratio_far = max(0.001, args.far_ratio)
    if args.target_ratio is not None:
        app.area_ratio_target = max(app.area_ratio_far, args.target_ratio)
    if args.near_ratio is not None:
        app.area_ratio_near = max(app.area_ratio_target, args.near_ratio)
    app.debug = args.debug
    app.run()


if __name__ == '__main__':
    main()
