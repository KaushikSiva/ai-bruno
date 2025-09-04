#!/usr/bin/env python3
# coding: utf-8
"""
Bruno Buddy (voice or text chat with LM Studio)

Features
- Listens via microphone (if available) and transcribes to text.
- Sends conversation to a local LM Studio server (OpenAI-compatible /chat/completions).
- Speaks responses with the existing Inworld TTS (audio_tts.TTSSpeaker) if enabled, and also prints.
- Say "stop" (or type stop/quit/exit) to end the session.

Run examples
  python3 bruno_surveillance/buddy.py --audio --voice Ashley
  python3 bruno_surveillance/buddy.py                # text-only fallback if mic not found

Env (optional)
  LLM_API_BASE  (default: http://localhost:1234/v1)
  LLM_MODEL     (default: lmstudio)
  LLM_API_KEY   (default: lm-studio)
  INWORLD_TTS_URL (default set in audio_tts.py)
  INWORLD_API_KEY (required for TTS)
"""
import os
import sys
import time
import json
import argparse
from typing import List, Dict, Optional

import requests

from utils import LOG
from audio_tts import TTSSpeaker


def _lmstudio_chat(messages: List[Dict[str, str]],
                   model: str = None,
                   api_base: str = None,
                   api_key: str = None,
                   temperature: float = 0.6,
                   max_tokens: int = 512) -> str:
    api_base = (api_base or os.environ.get('LLM_API_BASE', 'http://localhost:1234/v1')).rstrip('/')
    model = model or os.environ.get('LLM_MODEL', 'lmstudio')
    api_key = api_key or os.environ.get('LLM_API_KEY', 'lm-studio')
    url = f"{api_base}/chat/completions"
    payload = {
        'model': model,
        'messages': messages,
        'temperature': temperature,
        'max_tokens': max_tokens,
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }
    r = requests.post(url, json=payload, headers=headers, timeout=120)
    r.raise_for_status()
    data = r.json()
    return (data['choices'][0]['message']['content'] or '').strip()


def _groq_transcribe_wav(wav_bytes: bytes, model: Optional[str] = None, api_base: Optional[str] = None,
                         api_key: Optional[str] = None) -> str:
    """Send WAV bytes to Groq Whisper transcription and return text.
    Expects OpenAI-compatible /audio/transcriptions endpoint.
    """
    api_base = (api_base or os.environ.get('GROQ_API_BASE', 'https://api.groq.com/openai/v1')).rstrip('/')
    api_key = api_key or os.environ.get('GROQ_API_KEY')
    model = model or os.environ.get('GROQ_STT_MODEL', 'whisper-large-v3')
    if not api_key:
        return ''
    url = f"{api_base}/audio/transcriptions"
    files = {
        'file': ('audio.wav', wav_bytes, 'audio/wav'),
    }
    data = {
        'model': model,
        'response_format': 'json',
        # 'language': 'en',  # optionally hint language
    }
    headers = {
        'Authorization': f'Bearer {api_key}',
    }
    try:
        r = requests.post(url, headers=headers, data=data, files=files, timeout=120)
        r.raise_for_status()
        j = r.json()
        # OpenAI-compatible responses include 'text'
        txt = (j.get('text') or '').strip()
        return txt
    except Exception as e:
        LOG.warning(f'Groq STT failed: {e}')
        return ''


class STT:
    """Speech-to-text facade: Groq Whisper primary, Vosk fallback, then SR engines."""
    def __init__(self, device_index: int | None = None, vosk_model_path: str | None = None, use_groq: Optional[bool] = None):
        self._recognizer = None
        self._mic = None
        self._device_index = device_index
        self._vosk_model = None
        # Auto-enable Groq if key present unless explicitly disabled
        if use_groq is None:
            self._use_groq = bool(os.environ.get('GROQ_API_KEY'))
        else:
            self._use_groq = bool(use_groq)
        # Try Vosk model if provided
        if vosk_model_path:
            try:
                from vosk import Model  # type: ignore
                self._vosk_model = Model(vosk_model_path)
                LOG.info(f"🗣️  Vosk model loaded: {vosk_model_path}")
            except Exception as e:
                LOG.warning(f"Vosk model load failed: {e}")
                self._vosk_model = None
        # Some Pi images are noisy with ALSA/JACK prints; be robust and quiet
        try:
            import speech_recognition as sr  # type: ignore
            self._recognizer = sr.Recognizer()
            try:
                with _suppress_alsa():
                    # Conservative audio params to match many USB mics
                    self._mic = sr.Microphone(device_index=device_index, sample_rate=16000, chunk_size=1024)
                with self._mic as source:
                    # Tune recognition to be more responsive on Pi/USB mics
                    self._recognizer.dynamic_energy_threshold = True
                    self._recognizer.energy_threshold = int(os.environ.get('BUDDY_ENERGY_THRESHOLD', '200'))
                    self._recognizer.pause_threshold = float(os.environ.get('BUDDY_PAUSE_THRESHOLD', '0.6'))
                    self._recognizer.non_speaking_duration = float(os.environ.get('BUDDY_NON_SPEAKING', '0.3'))
                    self._recognizer.adjust_for_ambient_noise(source, duration=0.8)
                LOG.info(f'🎙️  Microphone ready (SpeechRecognition, device_index={self._mic.device_index})')
            except Exception as e:
                LOG.warning(f'No microphone: {e}')
                self._recognizer = None
                self._mic = None
        except Exception:
            self._recognizer = None
            self._mic = None
        # Announce STT backends
        primary = 'Groq Whisper' if self._use_groq else ('Vosk' if self._vosk_model else 'SpeechRecognition')
        LOG.info(f'STT primary: {primary} | fallback: ' + ('Vosk -> SR' if self._use_groq else ('SR' if self._vosk_model else 'SR (only)')))

    @property
    def available(self) -> bool:
        return self._recognizer is not None and self._mic is not None

    def listen_once(self, prompt: str = 'You (speak): ') -> str:
        if not self.available:
            # fallback to text prompt
            return input('You: ').strip()
        import speech_recognition as sr  # type: ignore
        print(prompt, end='', flush=True)
        try:
            with self._mic as source:
                with _suppress_alsa():
                    audio = self._recognizer.listen(source, timeout=10, phrase_time_limit=12)
            # Try Groq Whisper first if selected
            if self._use_groq:
                try:
                    wav = audio.get_wav_data(convert_rate=16000, convert_width=2)
                    txt = _groq_transcribe_wav(wav)
                    if txt:
                        return txt
                except Exception:
                    pass
            # Then Vosk (offline) if available
            if self._vosk_model is not None:
                try:
                    # Convert to 16k mono 16-bit PCM
                    pcm = audio.get_raw_data(convert_rate=16000, convert_width=2)
                    from vosk import KaldiRecognizer  # type: ignore
                    rec = KaldiRecognizer(self._vosk_model, 16000)
                    ok = rec.AcceptWaveform(pcm)
                    result_json = rec.Result() if ok else rec.PartialResult()
                    data = json.loads(result_json)
                    txt = (data.get('text') or '').strip()
                    if txt:
                        return txt
                except Exception:
                    pass
            # Then try cloud/offline engines via SpeechRecognition
            try:
                text = self._recognizer.recognize_google(audio)
                return text.strip()
            except Exception:
                try:
                    text = self._recognizer.recognize_sphinx(audio)
                    return text.strip()
                except Exception:
                    print("(didn't catch that)")
                    return ''
        except sr.WaitTimeoutError:
            print("(timeout)")
            return ''
        except Exception:
            print("(mic error)")
            return ''


from contextlib import contextmanager
import os as _os

@contextmanager
def _suppress_alsa():
    """Temporarily redirect C-level stderr to /dev/null to silence ALSA/JACK noise."""
    try:
        devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
        saved_stderr_fd = _os.dup(2)
        _os.dup2(devnull_fd, 2)
        _os.close(devnull_fd)
        yield
    finally:
        try:
            _os.dup2(saved_stderr_fd, 2)
            _os.close(saved_stderr_fd)
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser(description='Bruno Buddy (LM Studio voice/chat)')
    p.add_argument('--audio', action='store_true', help='Speak responses with TTS')
    p.add_argument('--voice', default=os.environ.get('BRUNO_AUDIO_VOICE', 'Ashley'))
    p.add_argument('--system', default='You are Bruno, a concise, friendly assistant. Keep replies brief and helpful.')
    p.add_argument('--mic-index', type=int, default=None, help='SpeechRecognition microphone device index')
    p.add_argument('--list-mics', action='store_true', help='List available microphones and exit')
    p.add_argument('--vosk-model', default=os.environ.get('VOSK_MODEL', None), help='Path to Vosk offline model (fallback)')
    p.add_argument('--wake', default=os.environ.get('BUDDY_WAKE', 'hey bruno'), help='Wake phrase; only respond when transcript contains this (default: "hey bruno")')
    args = p.parse_args()

    if args.list_mics:
        try:
            import speech_recognition as sr  # type: ignore
            names = sr.Microphone.list_microphone_names()
            print('Available microphones:')
            for i, name in enumerate(names):
                print(f'  [{i}] {name}')
        except Exception as e:
            print(f'Could not list microphones: {e}')
        return

    tts = TTSSpeaker(enabled=args.audio, voice=args.voice) if args.audio else None
    if tts:
        try:
            tts.start()
        except Exception as e:
            LOG.warning(f'TTS init failed: {e}')
            tts = None

    stt = STT(device_index=args.mic_index, vosk_model_path=args.vosk_model, use_groq=None)
    if not stt.available:
        LOG.info('Mic unavailable; falling back to keyboard input.')

    messages: List[Dict[str, str]] = [{'role': 'system', 'content': args.system}]
    wake_info = f" Say '{args.wake}' to talk." if args.wake else ''
    print(f"Buddy ready.{wake_info} Say or type 'stop' to end.\n")

    try:
        empty_in_a_row = 0
        while True:
            user_text = stt.listen_once()
            if not user_text:
                empty_in_a_row += 1
                if empty_in_a_row >= 3 and not stt.available:
                    # Already in text mode; keep looping
                    continue
                if empty_in_a_row >= 3 and stt.available:
                    # Offer quick keyboard fallback after repeated silence
                    typed = input('You (type, fallback): ').strip()
                    if typed:
                        user_text = typed
                        empty_in_a_row = 0
                    else:
                        continue
                continue
            empty_in_a_row = 0
            # Wake word gating
            if args.wake:
                low = user_text.lower()
                wake = args.wake.lower()
                if wake not in low:
                    # ignore until wake phrase detected
                    continue
                # Strip only the first occurrence (keeps rest of content)
                idx = low.find(wake)
                user_text = user_text[:idx] + user_text[idx+len(args.wake):]
                user_text = user_text.strip() or 'hello'
            print(f'You: {user_text}')
            if user_text.strip().lower() in ('stop', 'quit', 'exit'):
                print('Bye!')
                break

            messages.append({'role': 'user', 'content': user_text})
            try:
                reply = _lmstudio_chat(messages)
            except Exception as e:
                reply = f"[LM Studio error] {e}"
            messages.append({'role': 'assistant', 'content': reply})

            print(f'Buddy: {reply}\n')
            if tts:
                try:
                    tts.speak_sync(reply)
                except Exception:
                    pass
    except KeyboardInterrupt:
        print('\nBye!')
    finally:
        try:
            if tts:
                tts.stop(wait=False)
        except Exception:
            pass


if __name__ == '__main__':
    main()
