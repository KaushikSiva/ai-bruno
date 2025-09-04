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
from typing import List, Dict

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


class STT:
    """Simple speech-to-text facade: tries SpeechRecognition; else falls back to input()."""
    def __init__(self, device_index: int | None = None):
        self._recognizer = None
        self._mic = None
        try:
            import speech_recognition as sr  # type: ignore
            self._recognizer = sr.Recognizer()
            try:
                self._mic = sr.Microphone(device_index=device_index)
                with self._mic as source:
                    self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
                LOG.info(f'🎙️  Microphone ready (SpeechRecognition, device_index={self._mic.device_index})')
            except Exception as e:
                LOG.warning(f'No microphone: {e}')
                self._recognizer = None
                self._mic = None
        except Exception:
            self._recognizer = None
            self._mic = None

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
                audio = self._recognizer.listen(source, timeout=10, phrase_time_limit=12)
            # Try several engines in order
            try:
                text = self._recognizer.recognize_google(audio)
                return text.strip()
            except Exception:
                try:
                    text = self._recognizer.recognize_sphinx(audio)
                    return text.strip()
                except Exception:
                    return ''
        except sr.WaitTimeoutError:
            return ''
        except Exception:
            return ''


def main():
    p = argparse.ArgumentParser(description='Bruno Buddy (LM Studio voice/chat)')
    p.add_argument('--audio', action='store_true', help='Speak responses with TTS')
    p.add_argument('--voice', default=os.environ.get('BRUNO_AUDIO_VOICE', 'Ashley'))
    p.add_argument('--system', default='You are Bruno, a concise, friendly assistant. Keep replies brief and helpful.')
    p.add_argument('--mic-index', type=int, default=None, help='SpeechRecognition microphone device index')
    p.add_argument('--list-mics', action='store_true', help='List available microphones and exit')
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

    stt = STT(device_index=args.mic_index)
    if not stt.available:
        LOG.info('Mic unavailable; falling back to keyboard input.')

    messages: List[Dict[str, str]] = [{'role': 'system', 'content': args.system}]
    print('Buddy ready. Say or type "stop" to end.\n')

    try:
        while True:
            user_text = stt.listen_once()
            if not user_text:
                # ignore silence/timeouts
                continue
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
