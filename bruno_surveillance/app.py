#!/usr/bin/env python3
# coding: utf-8
"""
Bruno External Camera Surveillance (local captioner → LM Studio summary)
- Captions snapshots locally using captioner.py
- Collects captions for a fixed duration, then asks LM Studio (OpenAI-compatible) for a concise summary
- Ultrasonic-based safety with EMERGENCY reverse + optional exit
"""
import os, sys, time, signal, logging
from typing import Optional, Dict, List

# Ensure Hiwonder SDK path is available for submodules that import it
sys.path.append('/home/pi/MasterPi')

# Local imports
from utils import LOG, paths, load_env, get_env_int, get_env_float
from camera_setup import CameraManager
from ultrasonic import UltrasonicRGB
from snapshotter import Snapshotter
from caption_pipeline import caption_image
from lmstudio_client import summarize_captions_lmstudio
from audio_tts import TTSSpeaker
from robot_motion import MecanumWrapper

try:
    import cv2
    from PIL import Image
except Exception as e:
    raise RuntimeError(f"Missing OpenCV/Pillow: {e}")

# ----- Load .env (if present) -----
load_env()

PHOTO_INTERVAL = get_env_int('PHOTO_INTERVAL_SEC', 15)
SUMMARY_DELAY  = get_env_int('SUMMARY_DELAY_SEC', 120)

def _env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in ('1', 'true', 'yes', 'on')

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

    # Audio (off by default). Enable via .env BRUNO_AUDIO_ENABLED=1 or CLI --audio/--talk
    'audio_enabled':    False,
    'audio_voice':      os.environ.get('BRUNO_AUDIO_VOICE', 'alloy'),
}

class BrunoExternalCameraSurveillance:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.motion = MecanumWrapper(
            forward_speed=cfg['forward_speed'],
            turn_speed=cfg['turn_speed']
        )
        self.ultra = UltrasonicRGB()

        # Optional audio speaker (isolated thread)
        # Flag can come from env or CLI args; env wins if present.
        cli_audio = ('--audio' in sys.argv) or ('--talk' in sys.argv)
        audio_enabled = _env_bool('BRUNO_AUDIO_ENABLED', cfg.get('audio_enabled', False)) or cli_audio
        self.speaker = TTSSpeaker(enabled=audio_enabled, voice=cfg.get('audio_voice', 'alloy'))
        if audio_enabled:
            try:
                self.speaker.start()
                LOG.info('🔊 Audio TTS enabled')
            except Exception as e:
                LOG.warning(f'Audio init failed; continuing without TTS: {e}')

        # >>> camera now fully managed by CameraManager <<<
        self.cam = CameraManager(
            retry_attempts=cfg['camera_retry_attempts'],
            retry_delay=cfg['camera_retry_delay']
        )

        self.snapshotter = Snapshotter(cfg['photo_interval'])
        self.start_time = time.time()
        self.summary_due_at = self.start_time + SUMMARY_DELAY
        self.captions: List[Dict] = []

        self.emergency_start: Optional[float] = None
        self.emergency_reversed_once: bool = False
        self.frame_idx = 0

    def _open_camera(self) -> bool:
        LOG.info('🎥 Opening external camera...')
        return self.cam.open()

    def _do_snapshot_and_caption(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        img_path = paths.save_image_path('external_camera_photo')
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
        LOG.info('🤖 Bruno External Camera Surveillance (local captioner → LM Studio summary)')
        LOG.info(f'🗂  Images will be saved to: {paths.gpt_images}')
        LOG.info(f'🗂  Logs will be saved to: {paths.logs}/bruno.log')
        LOG.info(f'🕒 Snapshot interval: {self.cfg["photo_interval"]}s | Summary at: {SUMMARY_DELAY}s')

        # Greeting
        try:
            if getattr(self, 'speaker', None):
                self.speaker.say("Hey, I'm Bruno")
        except Exception:
            pass

        if not self._open_camera():
            LOG.error('❌ Cannot start without external camera')
            return

        running = True
        last_good_frame = None
        last_distance_cm: Optional[float] = None

        try:
            while running:
                self.frame_idx += 1

                # --- unified read via CameraManager ---
                ok, frame = self.cam.read()
                if ok and frame is not None:
                    last_good_frame = frame
                elif not ok:
                    LOG.warning('📹 Camera read failed; attempting reconnection...')
                    self.cam.reopen()

                # Snapshot cadence
                if self.snapshotter.due():
                    if last_good_frame is not None:
                        LOG.info('🛑 STOP for snapshot (pre-emptive)…')
                        self.motion.stop()
                        time.sleep(0.15)

                        # Try to get a fresh frame via the camera manager
                        fresh = self.cam.get_fresh_frame(max_attempts=3, settle_reads=3)
                        use_frame = fresh if fresh is not None else last_good_frame
                        self._do_snapshot_and_caption(use_frame)

                        if last_distance_cm is None or last_distance_cm > self.cfg['ultra_caution_cm']:
                            self.motion.forward()

                        time.sleep(0.2)
                    else:
                        LOG.warning('⚠️  Snapshot due, but no frame available yet (skipping).')

                # Ultrasonic logic
                d_cm = self.ultra.get_distance_cm()
                last_distance_cm = d_cm

                if d_cm is not None and d_cm <= self.cfg['ultra_danger_cm']:
                    # Enter or remain in EMERGENCY
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

                    # After 5s stuck: reverse one step (only once)
                    if (elapsed >= 5.0) and (not self.emergency_reversed_once):
                        LOG.warning('⏪ EMERGENCY >5s — reversing one step')
                        self.motion.reverse_burst(duration=0.6)
                        self.emergency_reversed_once = True

                    # After 10s stuck: send summary and exit (your original used ~30s)
                    if elapsed >= 30.0:
                        self._finish_with_summary_and_exit('prolonged emergency stop (>10s)')

                elif d_cm is not None and d_cm <= self.cfg['ultra_caution_cm']:
                    # Caution: avoidance
                    self.ultra.set_rgb(255, 180, 0)
                    if self.emergency_start is not None:
                        LOG.info('✅ Left EMERGENCY zone — timer reset')
                    self.emergency_start = None
                    self.emergency_reversed_once = False

                    if self.cfg['backup_time'] > 0:
                        self.motion.forward(duration=self.cfg['backup_time'])
                        self.motion.stop()

                    left = ((self.frame_idx // 60) % 2 == 0)
                    self.motion.turn_left(self.cfg['turn_time']) if left else self.motion.turn_right(self.cfg['turn_time'])
                    LOG.info(f"ULTRA AVOID ({d_cm:.1f} cm) {'LEFT' if left else 'RIGHT'}")

                else:
                    # Safe: drive forward
                    self.ultra.set_rgb(0, 255, 0)
                    if self.emergency_start is not None:
                        LOG.info('✅ Safe distance — emergency cleared')
                    self.emergency_start = None
                    self.emergency_reversed_once = False
                    self.motion.forward()

                # Time-based summary
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
            if getattr(self, 'speaker', None):
                # Best-effort graceful stop without blocking shutdown too long
                self.speaker.stop(wait=False)
        except Exception:
            pass
        try:
            self.cam.release()
        except Exception:
            pass
        try:
            self.ultra.set_rgb(0, 0, 0)
        except Exception:
            pass

RUNNER: Optional[BrunoExternalCameraSurveillance] = None

def _sig_handler(signum, frame):
    global RUNNER
    print('\n🛑 Ctrl-C received; stopping...')
    if RUNNER:
        RUNNER.shutdown()
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, _sig_handler)
    LOG.info('🤖 Bruno External Camera Surveillance (local captioner → LM Studio summary)')
    LOG.info('Press Ctrl+C to stop')
    RUNNER = BrunoExternalCameraSurveillance(CONFIG)
    RUNNER.run()
