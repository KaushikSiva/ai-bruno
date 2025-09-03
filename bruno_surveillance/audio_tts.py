import os
import io
import time
import queue
import threading
import tempfile
import subprocess
from typing import Optional, Callable

import requests

from utils import LOG, get_env_str


class TTSSpeaker:
    """
    Background TTS speaker using OpenAI's Audio Speech API.

    - Start a worker thread that consumes a queue of text to speak.
    - Uses OpenAI TTS (gpt-4o-mini-tts) to get WAV bytes.
    - Playback via simpleaudio if available, else falls back to system players (aplay/ffplay).

    This class is self-contained and optional. If API key is missing or any
    dependency fails, it degrades gracefully and logs warnings.
    """

    def __init__(self,
                 enabled: bool,
                 voice: str = 'alloy',
                 model: str = 'gpt-4o-mini-tts',
                 api_base: str = 'https://api.openai.com/v1',
                 backend: str = 'openai'):
        self.enabled = bool(enabled)
        self.voice = voice
        self.model = model
        self.api_base = api_base.rstrip('/')
        self.backend = (backend or 'openai').strip().lower()

        self.api_key = os.environ.get('OPENAI_API_KEY')
        self._local_speak: Optional[Callable[[str], None]] = None

        if self.enabled:
            if self.backend == 'openai':
                if not self.api_key:
                    LOG.warning('Audio backend openai selected, but OPENAI_API_KEY is missing. Disabling TTS.')
                    self.enabled = False
            elif self.backend == 'local':
                # Try to import a local speak function: module and func can be overridden via env
                mod_path = os.environ.get('BRUNO_AUDIO_LOCAL_MODULE', 'tts_voice.speak')
                func_name = os.environ.get('BRUNO_AUDIO_LOCAL_FUNC', 'speak')
                try:
                    module = __import__(mod_path, fromlist=[func_name])
                    self._local_speak = getattr(module, func_name)
                    LOG.info(f'Audio backend: local ({mod_path}.{func_name})')
                except Exception as e:
                    LOG.warning(f'Local audio backend import failed ({mod_path}.{func_name}): {e}. Falling back to openai.')
                    self.backend = 'openai'
                    if not self.api_key:
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
        url = f'{self.api_base}/audio/speech'
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        payload = {
            'model': self.model,
            'input': text,
            'voice': self.voice,
            'response_format': 'wav',
        }
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=60)
            r.raise_for_status()
            return r.content  # WAV bytes
        except Exception as e:
            LOG.warning(f'OpenAI TTS request failed: {e}')
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
