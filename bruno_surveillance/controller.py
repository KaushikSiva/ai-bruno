#!/usr/bin/env python3
# coding: utf-8
"""
Readable controller orchestrating the surveillance loop.
Splits configuration and safety logic into small modules.
"""
import os
import sys
import time
import signal
from typing import Optional, List, Dict

# Hiwonder SDK path for robot deps
sys.path.append('/home/pi/MasterPi')

from utils import LOG, paths
from settings import load_settings, Settings
from safety import SafetyController
from snapshotter import Snapshotter
from caption_pipeline import caption_image
from lmstudio_client import summarize_captions_lmstudio
from audio_tts import TTSSpeaker
from robot_motion import MecanumWrapper
from ultrasonic import UltrasonicRGB
from camera_shared import make_camera, read_or_reconnect

try:
    import cv2
    from PIL import Image
except Exception as e:
    raise RuntimeError(f"Missing OpenCV/Pillow: {e}")


class BrunoController:
    """Main runtime: camera → snapshot/caption → safety → summary."""

    def __init__(self, mode: str, audio_enabled: bool = False, audio_voice: str = 'alloy'):
        self.cfg: Settings = load_settings()
        self.mode = mode

        # Hardware + IO
        self.motion = MecanumWrapper(forward_speed=self.cfg.forward_speed, turn_speed=self.cfg.turn_speed)
        self.ultra = UltrasonicRGB()
        self.camera = make_camera(mode, self.cfg.camera_retry_attempts, self.cfg.camera_retry_delay)
        self.snapshotter = Snapshotter(self.cfg.photo_interval)
        self.safety = SafetyController(
            ultra_caution_cm=self.cfg.ultra_caution_cm,
            ultra_danger_cm=self.cfg.ultra_danger_cm,
            turn_time=self.cfg.turn_time,
            backup_time=self.cfg.backup_time,
        )

        # Captions buffer and timeline
        self.captions: List[Dict] = []
        self.start_time = time.time()
        self.summary_due_at = self.start_time + self.cfg.summary_delay
        self.frame_idx = 0

        # Optional speech
        self.speaker = None
        if audio_enabled:
            try:
                self.speaker = TTSSpeaker(enabled=True, voice=audio_voice)
                self.speaker.start()
                LOG.info('🔊 Audio TTS enabled')
            except Exception as e:
                LOG.warning(f'Audio init failed; continuing without TTS: {e}')

    # ----- High-level lifecycle -----
    def run(self):
        self._log_startup()
        self._greet()
        if not self.camera.open():
            LOG.error('❌ Cannot start without camera')
            return
        self._loop()

    def shutdown(self):
        LOG.info('Shutting down...')
        self.motion.stop()
        try:
            if self.speaker:
                self.speaker.stop(wait=False)
        except Exception:
            pass
        try:
            if self.camera:
                self.camera.release()
        except Exception:
            pass
        try:
            self.ultra.set_rgb(0, 0, 0)
        except Exception:
            pass

    # ----- Loop helpers -----
    def _loop(self):
        last_good_frame = None
        last_distance_cm: Optional[float] = None
        try:
            while True:
                self.frame_idx += 1

                # 1) Read a frame (with automatic reconnect on failure)
                last_good_frame = read_or_reconnect(self.camera, last_good_frame)

                # 2) Snapshot cadence (camera-agnostic)
                if self.snapshotter.due():
                    if last_good_frame is not None:
                        self._take_and_caption(last_good_frame)
                        if last_distance_cm is None or last_distance_cm > self.cfg.ultra_caution_cm:
                            self.motion.forward()
                    else:
                        LOG.warning('⚠️  Snapshot due, but no frame available yet (skipping).')

                d_cm = self.ultra.get_distance_cm()
                last_distance_cm = d_cm

                try:
                    self.safety.handle(d_cm, self.frame_idx, self.motion, self.ultra)
                except SystemExit as e:
                    self._finish_with_summary_and_exit(str(e))

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

    # ----- Actions -----
    def _take_and_caption(self, frame) -> None:
        LOG.info('🛑 STOP for snapshot (pre‑emptive)…')
        self.motion.stop()
        time.sleep(0.15)
        fresh = self.camera.get_fresh_frame(max_attempts=3, settle_reads=3)
        use_frame = fresh if fresh is not None else frame

        rgb = cv2.cvtColor(use_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        img_path = paths.save_image_path(f'{self.mode}_camera_photo')
        image.save(str(img_path))
        self.snapshotter.mark()
        LOG.info(f'💾 Snapshot saved: {img_path}')

        cap_text = caption_image(str(img_path))
        LOG.info(f'📝 Caption: {cap_text}')
        try:
            if self.speaker:
                self.speaker.say(cap_text)
        except Exception:
            pass
        ts = time.strftime('%H:%M:%S')
        paths.safe_write_text(paths.sidecar_txt_path(img_path), cap_text + '\n')
        self.captions.append({'timestamp': ts, 'path': str(img_path), 'caption': cap_text})
        time.sleep(0.2)

    def _finish_with_summary_and_exit(self, reason: str):
        LOG.warning(f'⏹️  Triggering early summary due to: {reason}')
        summary = summarize_captions_lmstudio(self.captions)
        print('\n' + '=' * 80)
        print('🧾 SUMMARY (LM Studio):')
        print(summary)
        print('=' * 80 + '\n')
        try:
            if self.speaker:
                self.speaker.say(summary)
                self.speaker.wait_idle(timeout=45.0)
        except Exception:
            pass
        self.shutdown()
        sys.exit(0)

    # ----- Small utilities -----
    def _log_startup(self):
        LOG.info('🤖 Bruno Dual-Mode Surveillance (local captioner → LM Studio summary)')
        LOG.info(f'🗂  Images: {paths.gpt_images} | Logs: {paths.logs}/bruno.log')
        LOG.info(f'🕒 Snapshot interval: {self.cfg.photo_interval}s | Summary at: {self.cfg.summary_delay}s')

    def _greet(self):
        try:
            if self.speaker:
                self.speaker.say("Hey, I'm Bruno")
        except Exception:
            pass

    # (camera read helper moved to camera_shared.read_or_reconnect)


# ----- Entrypoint used by app.py -----
RUNNER: Optional['BrunoController'] = None


def _sig_handler(signum, frame):
    global RUNNER
    print('\n🛑 Ctrl‑C received; stopping...')
    if RUNNER:
        RUNNER.shutdown()
    sys.exit(0)


def run(mode: str, audio_enabled: bool = False, audio_voice: str = 'alloy'):
    signal.signal(signal.SIGINT, _sig_handler)
    global RUNNER
    RUNNER = BrunoController(mode=mode, audio_enabled=audio_enabled, audio_voice=audio_voice)
    RUNNER.run()
