import os
import io
import time
import queue
import threading
import tempfile
import subprocess
from typing import Optional, Callable

import json
import base64
import wave

import requests

from utils import LOG


class TTSSpeaker:
    """Minimal TTS speaker using Inworld (voice Ashley)."""

    def __init__(self, enabled: bool, voice: str = 'Ashley'):
        self.enabled = bool(enabled)
        self.voice = voice or 'Ashley'

        self.inworld_url = os.environ.get('INWORLD_TTS_URL', 'https://api.inworld.ai/tts/v1/voice:stream').strip()
        self.inworld_api_key = os.environ.get('INWORLD_API_KEY', '').strip()

        if self.enabled and not self.inworld_api_key:
            LOG.warning('INWORLD_API_KEY not set. Disabling TTS.')
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

    def speak_sync(self, text: str) -> None:
        """Synthesize and play immediately on the calling (main) thread."""
        if not self.enabled:
            return
        t = (text or '').strip()
        if not t:
            return
        try:
            audio = self._synthesize_tts(t)
            if audio:
                self._play_audio(audio)
        except Exception as e:
            LOG.warning(f'TTS sync playback failed: {e}')

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
                audio = self._synthesize_tts(text)
                if audio:
                    self._play_audio(audio)
            except Exception as e:
                LOG.warning(f'TTS playback failed: {e}')
            finally:
                self._q.task_done()
        LOG.info('🔇 TTS worker stopped')

    def _synthesize_tts(self, text: str) -> Optional[bytes]:
        """Simple Inworld TTS: Ashley voice, fixed payload, stream chunks, return WAV bytes."""
        if not self.inworld_api_key or requests is None:
            return None
        url = self.inworld_url or 'https://api.inworld.ai/tts/v1/voice:stream'
        headers = {
            'Authorization': f'Basic {self.inworld_api_key}',
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream',
        }
        payload = {
            'text': text,
            'voiceId': self.voice or 'Ashley',
            'modelId': 'inworld-tts-1',
            'audio_config': {
                'audio_encoding': 'LINEAR16',
                'sample_rate_hertz': 48000,
            }
        }
        try:
            r = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)
            r.raise_for_status()
            raw_pcm = io.BytesIO()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    # Lines may be prefixed with 'data:' (SSE)
                    if isinstance(line, bytes):
                        s = line.decode('utf-8', errors='ignore').strip()
                    else:
                        s = str(line).strip()
                    if not s:
                        continue
                    if s.startswith('data:'):
                        s = s[len('data:'):].strip()
                    if not s or s[0] != '{':
                        continue
                    chunk = json.loads(s)
                    audio_b64 = (
                        (chunk.get('result') or {}).get('audioContent')
                        or chunk.get('audioContent')
                        or (chunk.get('tts') or {}).get('audioContent')
                    )
                    if not audio_b64:
                        continue
                    audio_bytes = base64.b64decode(audio_b64)
                    # Many chunks are small WAVs; strip header if present
                    raw_pcm.write(audio_bytes[44:] if len(audio_bytes) > 44 else audio_bytes)
                except Exception:
                    continue
            pcm = raw_pcm.getvalue()
            if not pcm:
                LOG.warning('Inworld TTS produced no audio')
                return None
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(pcm)
            return buf.getvalue()
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
