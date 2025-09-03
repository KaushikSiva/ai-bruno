#!/usr/bin/env python3
# coding: utf-8
"""
Bruno Dual-Mode Surveillance (Local Captioner → LM Studio Summary)
- Switch between 'builtin' and 'external' camera via --mode or CAM_MODE env
- Modularized into small files
"""
import os, sys, time, signal, argparse
from typing import Optional, Dict, List

# Hiwonder SDK path for robot deps
sys.path.append('/home/pi/MasterPi')

from utils import LOG, paths, load_env, get_env_int, get_env_float
from camera_setup import find_working_external_camera
from ultrasonic import UltrasonicRGB
from snapshotter import Snapshotter
from caption_pipeline import caption_image
from lmstudio_client import summarize_captions_lmstudio
from audio_tts import TTSSpeaker
from robot_motion import MecanumWrapper
from ultrasonic import UltrasonicRGB
from camera_manager import make_camera  # factory for builtin/external

try:
    import cv2
    from PIL import Image
except Exception as e:
    raise RuntimeError(f"Missing OpenCV/Pillow: {e}")

# ----- Load .env (if present) -----
load_env()

PHOTO_INTERVAL = get_env_int('PHOTO_INTERVAL_SEC', 15)
SUMMARY_DELAY  = get_env_int('SUMMARY_DELAY_SEC', 120)

CONFIG: Dict = {
    'camera_retry_attempts': 3,
    'camera_retry_delay': 2.0,

    'ultra_caution_cm': get_env_float('ULTRA_CAUTION_CM', 50.0),
    'ultra_danger_cm':  get_env_float('ULTRA_DANGER_CM', 25.0),
    'forward_speed':    get_env_int('BRUNO_SPEED', 40),
    'turn_speed':       get_env_int('BRUNO_TURN_SPEED', 40),
    'turn_time':        get_env_float('BRUNO_TURN_TIME', 0.5),
    'backup_time':      get_env_float('BRUNO_BACKUP_TIME', 0.0),

    'photo_interval':   PHOTO_INTERVAL,
}

class BrunoSurveillance:
    def __init__(self, cfg: Dict, mode: str):
        self.cfg = cfg
        self.mode = mode
        self.motion = MecanumWrapper(
            forward_speed=cfg['forward_speed'],
            turn_speed=cfg['turn_speed']
        )
        self.ultra = UltrasonicRGB()
        self.cap = None
        self.camera_info = None

        self.snapshotter = Snapshotter(cfg['photo_interval'])
        self.start_time = time.time()
        self.summary_due_at = self.start_time + cfg['summary_delay']
        self.captions: List[Dict] = []
        self.emergency_start: Optional[float] = None
        self.emergency_reversed_once: bool = False
        self.frame_idx = 0

    def _open_camera(self) -> bool:
        LOG.info('🎥 Opening external camera...')
        for attempt in range(self.cfg['camera_retry_attempts']):
            LOG.info(f'📹 Camera connection attempt {attempt + 1}...')
            self.cap, self.camera_info = find_working_external_camera()
            if self.cap:
                LOG.info(f"✅ External camera connected: {self.camera_info['path']}")
                return True
            if attempt < self.cfg['camera_retry_attempts'] - 1:
                LOG.info(f"⏳ Retrying in {self.cfg['camera_retry_delay']} seconds...")
                time.sleep(self.cfg['camera_retry_delay'])
        LOG.error('❌ Failed to connect external camera')
        return False

    def _do_snapshot_and_caption(self, frame) -> None:
        try:
            frame_hash = hash(frame.tobytes())
            LOG.info(f"📸 Snapshot frame hash: {frame_hash}")
        except Exception:
            pass

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        img_path = paths.save_image_path(f'{self.mode}_camera_photo')
        image.save(str(img_path))
        self.snapshotter.mark()
        LOG.info(f'💾 Snapshot saved: {img_path}')

        cap_text = caption_image(str(img_path))
        LOG.info(f'📝 Caption: {cap_text}')

        # Speak caption (non-blocking, background thread)
        try:
            if getattr(self, 'speaker', None):
                self.speaker.say(cap_text)
        except Exception:
            pass

        ts = time.strftime('%H:%M:%S')
        txt_path = paths.sidecar_txt_path(img_path)
        paths.safe_write_text(txt_path, cap_text + '\n')
        self.captions.append({'timestamp': ts, 'path': str(img_path), 'caption': cap_text})

    def _finish_with_summary_and_exit(self, reason: str):
        LOG.warning(f'⏹️  Triggering early summary due to: {reason}')
        summary = summarize_captions_lmstudio(self.captions)
        print('\n' + '=' * 80)
        print('🧾 SUMMARY (LM Studio):')
        print(summary)
        print('=' * 80 + '\n')

        # Speak final summary then exit; give a brief window to finish
        try:
            if getattr(self, 'speaker', None):
                self.speaker.say(summary)
                self.speaker.wait_idle(timeout=45.0)
        except Exception:
            pass
        self.shutdown()
        sys.exit(0)

    def _maybe_finish_with_summary(self):
        if time.time() >= self.summary_due_at:
            self._finish_with_summary_and_exit('time window reached')

    def run(self):
        LOG.info('🤖 Bruno Dual-Mode Surveillance (local captioner → LM Studio summary)')
        LOG.info(f'🗂  Images: {paths.gpt_images} | Logs: {paths.logs}/bruno.log')
        LOG.info(f'🕒 Snapshot interval: {self.cfg["photo_interval"]}s | Summary at: {self.cfg["summary_delay"]}s')

        # Greeting
        try:
            if getattr(self, 'speaker', None):
                self.speaker.say("Hey, I'm Bruno")
        except Exception:
            pass

        if not self._open_camera():
            LOG.error('❌ Cannot start without camera')
            return

        running = True
        last_good_frame = None
        last_distance_cm: Optional[float] = None

        try:
            while running:
                self.frame_idx += 1
                frame = None

                # Grab frame
                if self.cap and self.cap.isOpened():
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        LOG.warning('📹 Camera read failed; attempting reconnection...')
                        self.cap.release()
                        time.sleep(1)
                        self._open_camera()
                    else:
                        last_good_frame = frame

                # Snapshot cadence
                if self.snapshotter.due():
                    if last_good_frame is not None:
                        LOG.info('🛑 STOP for snapshot (pre‑emptive)…')
                        self.motion.stop()
                        time.sleep(0.15)

                        fresh = None
                        if self.cap and self.cap.isOpened():
                            for _ in range(3):
                                self.cap.read(); time.sleep(0.02)
                            for _ in range(3):
                                ok, fresh = self.cap.read()
                                if ok and fresh is not None:
                                    break
                                time.sleep(0.03)
                        use_frame = fresh if fresh is not None else last_good_frame
                        self._do_snapshot_and_caption(use_frame)

                        if last_distance_cm is None or last_distance_cm > self.cfg['ultra_caution_cm']:
                            self.motion.forward()

                        time.sleep(0.2)
                    else:
                        LOG.info('⚠️ Not resuming - obstacle detected')
                    
                    time.sleep(0.1)

                d_cm = self.ultra.get_distance_cm()
                last_distance_cm = d_cm

                if d_cm is not None and d_cm <= self.cfg['ultra_danger_cm']:
                    if self.emergency_start is None:
                        self.emergency_start = time.time()
                        self.emergency_reversed_once = False
                        LOG.warning(f'EMERGENCY STOP ({d_cm:.1f} cm) — timer started')
                    else:
                        LOG.warning(f'EMERGENCY STOP ({d_cm:.1f} cm) — ongoing')

                    self.ultra.set_rgb(255, 0, 0)
                    self.motion.stop()
                    time.sleep(0.02)

                    elapsed = time.time() - self.emergency_start
                    if (elapsed >= 5.0) and (not self.emergency_reversed_once):
                        LOG.warning('⏪ EMERGENCY >5s — reversing one step')
                        self.motion.reverse_burst(duration=0.6)
                        self.emergency_reversed_once = True

                    if elapsed >= 30.0:
                        self._finish_with_summary_and_exit('prolonged emergency stop (>30s)')

                elif d_cm is not None and d_cm <= self.cfg['ultra_caution_cm']:
                    self.ultra.set_rgb(255, 180, 0)
                    if self.emergency_start is not None:
                        LOG.info('✅ Left EMERGENCY zone — timer reset')
                    self.emergency_start = None
                    self.emergency_reversed_once = False

                    if self.cfg['backup_time'] > 0:
                        self.motion.forward(duration=self.cfg['backup_time'])
                        self.motion.stop()

                    left = ((self.frame_idx // 60) % 2 == 0)
                    if left: self.motion.turn_left(self.cfg['turn_time'])
                    else:    self.motion.turn_right(self.cfg['turn_time'])

                else:
                    self.ultra.set_rgb(0, 255, 0)
                    if self.emergency_start is not None:
                        LOG.info('✅ Safe distance — emergency cleared')
                    self.emergency_start = None
                    self.emergency_reversed_once = False
                    self.motion.forward()

                # Time‑based summary
                self._maybe_finish_with_summary()
                time.sleep(0.03)

        except KeyboardInterrupt:
            LOG.info('Interrupted by user')
            self.shutdown()
        except SystemExit:
            raise
        except Exception as e:
            LOG.error(f'Unexpected error: {e}')
            self.shutdown()

    def shutdown(self):
        LOG.info('Shutting down...')
        self.motion.stop()
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        try:
            self.ultra.set_rgb(0, 0, 0)
        except Exception:
            pass

RUNNER: Optional[BrunoExternalCameraSurveillance] = None

def _sig_handler(signum, frame):
    global RUNNER
    print('\n🛑 Ctrl‑C received; stopping...')
    if RUNNER:
        RUNNER.shutdown()
    sys.exit(0)

if __name__ == '__main__':
    main()
