import os
import io
import time
import queue
import threading
import tempfile
import subprocess
from typing import Optional, Callable
from contextlib import contextmanager

import json
import base64
import wave

import requests

from bruno_core.config.env import get_env_str
from bruno_core.logging.setup import LOG


@contextmanager
def _suppress_alsa():
    """Raspberry Pi optimized ALSA/JACK/Audio error suppression."""
    try:
        # Set environment variables to suppress all audio backend verbosity
        old_env = {}
        audio_env_vars = {
            # ALSA suppression - Pi specific (but preserve PulseAudio preference)
            'ALSA_PCM_CARD': '0',
            'ALSA_PCM_DEVICE': '0', 
            'ALSA_LOG_LEVEL': '0',
            'ALSA_PLUGIN_DIR': '/usr/lib/arm-linux-gnueabihf/alsa-lib',
            'ALSA_MIXER_SIMPLE': '1',
            # JACK suppression - Pi specific
            'JACK_NO_START_SERVER': '1',
            'JACK_NO_AUDIO_RESERVATION': '1', 
            'JACK_SILENCE_MESSAGES': '1',
            'JACK_DEFAULT_SERVER': 'dummy',
            'JACK_DRIVER': 'dummy',
            # PulseAudio configuration - prefer PulseAudio over ALSA direct
            'PULSE_LATENCY_MSEC': '30',
            'PA_ALSA_PLUGHW': '1',
            # OSS suppression
            'OSS_AUDIODEV': '/dev/null',
            'OSS_MIXERDEV': '/dev/null',
            # Pi audio hardware specific
            'AUDIODEV': '/dev/null',
            'AUDIODRIVER': 'pulse',  # Prefer PulseAudio
            'SDL_AUDIODRIVER': 'pulse',  # Use PulseAudio for SDL
            # Additional Pi suppressions
            'LIBASOUND_DEBUG': '0',
            'ALSA_PERIOD_TIME': '0',
            'ALSA_BUFFER_TIME': '0',
            # Suppress specific ALSA error types
            'ALSA_PCM_STREAM': '0',
            'ALSA_RAWMIDI_STREAM': '0'
        }
        
        for key, value in audio_env_vars.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        # Redirect file descriptors
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        
        # Redirect both stdout and stderr to suppress audio messages
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        
        yield
        
    finally:
        try:
            # Restore file descriptors
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            
            # Restore environment variables
            for key, old_value in old_env.items():
                if old_value is not None:
                    os.environ[key] = old_value
                elif key in os.environ:
                    del os.environ[key]
        except Exception:
            pass


class TTSSpeaker:
    """Minimal TTS speaker using Inworld (voice Dominoux)."""

    def __init__(self, enabled: bool, voice: str = 'Dominoux'):
        self.enabled = bool(enabled)
        self.voice = voice or 'Dominoux'

        self.inworld_url = get_env_str('INWORLD_TTS_URL').strip()
        self.inworld_api_key = get_env_str('INWORLD_API_KEY').strip()

        # Check for fallback TTS options if Inworld API key not available
        self.use_fallback_tts = False
        self.fallback_tts_cmd = None
        
        if self.enabled and not self.inworld_api_key:
            # Try to find system TTS as fallback
            fallback_options = [
                ['espeak', '-s', '150'],  # eSpeak with moderate speed
                ['espeak-ng', '-s', '150'],
                ['festival', '--tts'],
                ['pico2wave', '-l', 'en-US', '-w', '/dev/stdout'],
                ['flite']
            ]
            
            for cmd in fallback_options:
                try:
                    result = subprocess.run([cmd[0], '--version'], capture_output=True, timeout=2)
                    if result.returncode == 0:
                        self.use_fallback_tts = True
                        self.fallback_tts_cmd = cmd
                        LOG.info(f'Using fallback TTS: {cmd[0]}')
                        break
                except (subprocess.SubprocessError, FileNotFoundError):
                    continue
            
            if not self.use_fallback_tts:
                LOG.warning('INWORLD_API_KEY not set and no fallback TTS found. Disabling TTS.')
                self.enabled = False

        self._q: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._interrupt = threading.Event()
        self._current_play = None  # simpleaudio PlayObject
        self._current_proc = None  # subprocess.Popen

        # Optional in-memory player (suppress audio backend errors during import)
        with _suppress_alsa():
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

    def interrupt(self):
        """Request immediate stop of any current playback and clear queue."""
        try:
            self._interrupt.set()
            # Best-effort immediate stop
            if self._current_play is not None:
                try:
                    self._current_play.stop()
                except Exception:
                    pass
            if self._current_proc is not None:
                try:
                    self._current_proc.terminate()
                except Exception:
                    pass
            # Drain any queued phrases
            try:
                while not self._q.empty():
                    self._q.get_nowait()
                    self._q.task_done()
            except Exception:
                pass
        except Exception:
            pass

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
                self._interrupt.clear()
                audio = self._synthesize_tts(text)
                if audio:
                    self._play_audio(audio)
            except Exception as e:
                LOG.warning(f'TTS playback failed: {e}')
            finally:
                self._q.task_done()
        LOG.info('🔇 TTS worker stopped')

    def _synthesize_tts(self, text: str) -> Optional[bytes]:
        """TTS synthesis with Inworld API or fallback to system TTS."""
        # Try Inworld API first
        if self.inworld_api_key and requests is not None:
            return self._inworld_tts(text)
        
        # Use fallback system TTS
        if self.use_fallback_tts and self.fallback_tts_cmd:
            return self._fallback_tts(text)
            
        return None
    
    def _inworld_tts(self, text: str) -> Optional[bytes]:
        """Inworld TTS synthesis."""
        url = self.inworld_url
        headers = {
            'Authorization': f'Basic {self.inworld_api_key}',
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream',
        }
        payload = {
            'text': text,
            'voiceId': self.voice or 'Dominoux',
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
    
    def _fallback_tts(self, text: str) -> Optional[bytes]:
        """Fallback TTS using system commands like espeak."""
        if not self.fallback_tts_cmd:
            return None
            
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
                # Prepare command based on TTS system
                cmd = self.fallback_tts_cmd.copy()
                
                if 'espeak' in cmd[0]:
                    # espeak -w output.wav "text"
                    cmd.extend(['-w', tmp.name, text])
                elif 'pico2wave' in cmd[0]:
                    # pico2wave -w output.wav "text"
                    cmd.extend(['-w', tmp.name, text])
                elif 'festival' in cmd[0]:
                    # echo "text" | festival --tts
                    cmd = ['bash', '-c', f'echo "{text}" | festival --tts --pipe']
                    # This won't produce WAV, skip for now
                    return None
                else:
                    # Generic: command "text" > output.wav
                    return None
                
                # Execute TTS command with suppression
                with _suppress_alsa():
                    result = subprocess.run(cmd, capture_output=True, timeout=10)
                
                if result.returncode == 0 and tmp.name:
                    # Read the generated WAV file
                    with open(tmp.name, 'rb') as f:
                        return f.read()
                else:
                    LOG.warning(f'Fallback TTS failed: {result.stderr}')
                    return None
                    
        except Exception as e:
            LOG.warning(f'Fallback TTS error: {e}')
            return None

    def _play_audio(self, wav_bytes: bytes):
        # Prefer simpleaudio if available (suppress audio backend errors)
        if self._sa is not None:
            import wave
            with _suppress_alsa():
                with io.BytesIO(wav_bytes) as bio:
                    with wave.open(bio, 'rb') as wf:
                        audio_data = wf.readframes(wf.getnframes())
                        obj = self._sa.WaveObject(audio_data, wf.getnchannels(), wf.getsampwidth(), wf.getframerate())
                        play = obj.play()
                        self._current_play = play
                        # Busy wait but allow early stop
                        while play.is_playing():
                            if self._stop.is_set() or self._interrupt.is_set():
                                try:
                                    play.stop()
                                except Exception:
                                    pass
                                break
                            time.sleep(0.05)
                self._current_play = None
                self._interrupt.clear()
                return

        # Fallback: write to temp WAV and invoke system player with suppressed audio errors
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp:
            tmp.write(wav_bytes)
            tmp.flush()
            
            # Set Pi system volume and audio device (suppress ALSA/JACK errors)
            with _suppress_alsa():
                try:
                    # Pi-specific volume control attempts
                    for vol_cmd in [
                        ["amixer", "set", "Master", "100%"],
                        ["amixer", "set", "PCM", "100%"],
                        ["amixer", "-c", "0", "set", "Master", "100%"]
                    ]:
                        try:
                            subprocess.run(vol_cmd, capture_output=True, timeout=3)
                            break
                        except:
                            continue
                except:
                    pass
            
            # Try Pi-optimized audio players with PulseAudio priority
            audio_commands = [
                # PulseAudio (preferred - usually works best on Pi)
                ["paplay", tmp.name],
                ["paplay", "--volume", "65536", tmp.name],     # Max volume
                # Pi-specific ALSA commands (fallback)
                ["aplay", "-q", "-D", "pulse", tmp.name],      # ALSA->PulseAudio bridge
                ["aplay", "-q", "-D", "default", tmp.name],    # ALSA default device
                ["aplay", "-q", "-D", "hw:0,0", tmp.name],     # Pi hardware device
                ["aplay", "-q", "-D", "plughw:0,0", tmp.name], # Pi plugin hardware
                ["aplay", "-q", tmp.name],                     # ALSA auto-detect
                # Final fallback
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "panic", "-volume", "100", tmp.name]
            ]
            
            for cmd in audio_commands:
                try:
                    with _suppress_alsa():
                        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        self._current_proc = proc
                    
                    # Poll with interrupt support (no suppression for responsiveness)
                    while proc.poll() is None:
                        if self._stop.is_set() or self._interrupt.is_set():
                            try:
                                proc.terminate()
                            except Exception:
                                pass
                            break
                        time.sleep(0.05)
                    self._current_proc = None
                    self._interrupt.clear()
                    return
                except FileNotFoundError:
                    continue
                except Exception:
                    continue
            LOG.warning('No audio player found (simpleaudio/aplay/ffplay). Skipping playback.')
