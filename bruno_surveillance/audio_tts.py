import os
import io
import time
import queue
import threading
import tempfile
import subprocess
from typing import Optional, Callable

import requests

from utils import LOG


class TTSSpeaker:
    """
    Background TTS speaker (Inworld or local backend).

    - Worker thread consumes a queue of text to speak.
    - Inworld backend: sends text to an HTTP endpoint (config via env), returns WAV bytes.
    - Local backend: imports a local `speak(text)` function (e.g., tts_voice.speak).
    - Playback via simpleaudio if available, else falls back to aplay/ffplay.
    """

    def __init__(self,
                 enabled: bool,
                 voice: str = 'default',
                 backend: str = 'inworld'):
        self.enabled = bool(enabled)
        self.voice = voice
        self.backend = (backend or 'inworld').strip().lower()

        # Inworld HTTP settings
        self.inworld_url = os.environ.get('INWORLD_TTS_URL', '').strip()
        self.inworld_api_key = os.environ.get('INWORLD_API_KEY', '').strip()

        # Optional local function
        self._local_speak: Optional[Callable[[str], None]] = None

        if self.enabled:
            if self.backend == 'local':
                mod_path = os.environ.get('BRUNO_AUDIO_LOCAL_MODULE', 'tts_voice.speak')
                func_name = os.environ.get('BRUNO_AUDIO_LOCAL_FUNC', 'speak')
                try:
                    module = __import__(mod_path, fromlist=[func_name])
                    self._local_speak = getattr(module, func_name)
                    LOG.info(f'Audio backend: local ({mod_path}.{func_name})')
                except Exception as e:
                    LOG.warning(f'Local audio backend import failed ({mod_path}.{func_name}): {e}. Disabling TTS.')
                    self.enabled = False
            else:
                # inworld backend by default
                if not self.inworld_url:
                    LOG.warning('INWORLD_TTS_URL not set. Disabling TTS.')
                    self.enabled = False

        self._q: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Optional in-memory player
        try:
            import simpleaudio  # type: ignore
            self._sa = simpleaudio
        except Exception:
            self._sa = None

    def start(self):
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name='TTSSpeaker', daemon=True)
        self._thread.start()

    def say(self, text: str):
        if not self.enabled:
            return
        t = (text or '').strip()
        if not t:
            return
        # Keep captions concise to avoid long latency bursts (optional cap)
        if len(t) > 800:
            t = t[:780] + '…'
        self._q.put(t)

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        """Block until the queue empties (best-effort). Returns True if empty."""
        if not self.enabled:
            return True
        start = time.time()
        while not self._q.empty():
            if timeout is not None and (time.time() - start) >= timeout:
                return False
            time.sleep(0.05)
        # Give the player a small grace window to finish the current clip
        if timeout is not None:
            time.sleep(min(0.3, max(0.0, timeout - (time.time() - start))))
        else:
            time.sleep(0.3)
        return self._q.empty()

    def stop(self, wait: bool = False, timeout: Optional[float] = 5.0):
        if not self.enabled:
            return
        if wait:
            self.wait_idle(timeout)
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    # ---------------- Internal helpers ----------------
    def _worker(self):
        LOG.info('🔊 TTS worker started')
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if self.backend == 'local' and self._local_speak is not None:
                    # Delegate playback to local function
                    self._local_speak(text)
                else:
                    audio = self._synthesize_tts(text)
                    if audio:
                        self._play_audio(audio)
            except Exception as e:
                LOG.warning(f'TTS playback failed: {e}')
            finally:
                self._q.task_done()
        LOG.info('🔇 TTS worker stopped')

    def _synthesize_tts(self, text: str) -> Optional[bytes]:
        """Call Inworld TTS HTTP endpoint and return WAV bytes.

        Expected env/config:
        - INWORLD_TTS_URL: full URL to a TTS endpoint that accepts JSON {"text", "voice"}
          and returns audio/wav bytes. Authorization header (Bearer) optional.
        - INWORLD_API_KEY: if set, sent as Bearer token.
        """
        if not self.inworld_url:
            return None
        headers = {'Content-Type': 'application/json'}
        if self.inworld_api_key:
            headers['Authorization'] = f'Bearer {self.inworld_api_key}'
        payload = {'text': text, 'voice': self.voice}
        try:
            r = requests.post(self.inworld_url, json=payload, headers=headers, timeout=60)
            r.raise_for_status()
            # Prefer binary content; if JSON with base64 provided, user can adapt endpoint or extend here.
            return r.content
        except Exception as e:
            LOG.warning(f'Inworld TTS request failed: {e}')
            return None

    def _play_audio(self, wav_bytes: bytes):
        # Prefer simpleaudio if available (in‑process playback)
        if self._sa is not None:
            import wave
            with io.BytesIO(wav_bytes) as bio:
                with wave.open(bio, 'rb') as wf:
                    audio_data = wf.readframes(wf.getnframes())
                    obj = self._sa.WaveObject(audio_data, wf.getnchannels(), wf.getsampwidth(), wf.getframerate())
                    play = obj.play()
                    # Busy wait but allow early stop
                    while play.is_playing():
                        if self._stop.is_set():
                            break
                        time.sleep(0.05)
            return

        # Fallback: write to temp WAV and invoke system player (aplay/ffplay)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp:
            tmp.write(wav_bytes)
            tmp.flush()
            for cmd in (["aplay", "-q", tmp.name], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", tmp.name]):
                try:
                    subprocess.run(cmd, timeout=120)
                    return
                except FileNotFoundError:
                    continue
                except Exception:
                    continue
            LOG.warning('No audio player found (simpleaudio/aplay/ffplay). Skipping playback.')
