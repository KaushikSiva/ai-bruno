#!/usr/bin/env python3
# coding: utf-8
"""
Bruno Buddy Enhanced - Wake-up System with Dynamic Greetings & Session Memory

Features:
- Sleep mode with head facing down
- Wake detection with "Hey Bruno!" trigger
- 30+ dynamic greetings on first wake-up per session
- Session memory for follow-up conversations
- Head movement integration
- Memory resets between sleep cycles

Usage:
  python3 buddy_test.py --audio --voice Ashley
  python3 buddy_test.py --wake "hey bruno"
"""

import os
import sys
import time
import json
import re
import random
import argparse
from typing import List, Dict, Optional, Any

import requests

from utils import LOG
from audio_tts import TTSSpeaker


class GreetingManager:
    """Manages 30+ dynamic greetings loaded from JSON with rotation to avoid repetition."""
    
    def __init__(self):
        self.greetings_data = self._load_greetings()
        self.greetings = self.greetings_data['greetings']
        self.responses = self.greetings_data['responses']
        self.used_greetings = []
        
    def _load_greetings(self) -> Dict[str, Any]:
        """Load greetings from JSON file."""
        assets_path = os.path.join(os.path.dirname(__file__), 'assets', 'greetings.json')
        try:
            with open(assets_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            LOG.warning(f"Failed to load greetings from {assets_path}: {e}")
            # Fallback to minimal greetings if file missing
            return {
                "greetings": [
                    {"text": "Well hello there! How are you today?", "offers": []},
                    {"text": "Hey! Good to see you again!", "offers": []},
                    {"text": "Hi there! What's new?", "offers": []}
                ],
                "responses": {}
            }
        
    def get_random_greeting(self) -> Dict[str, Any]:
        """Get a random greeting dict, avoiding recently used ones."""
        # If we've used all greetings, reset the rotation
        if len(self.used_greetings) >= len(self.greetings):
            self.used_greetings = []
            
        # Get unused greetings
        available = [g for g in self.greetings if g['text'] not in self.used_greetings]
        
        # Pick random greeting
        greeting = random.choice(available)
        self.used_greetings.append(greeting['text'])
        
        return greeting
        
    def get_response_for_offer(self, offer_type: str) -> Optional[str]:
        """Get a random response for a specific offer type."""
        if offer_type in self.responses:
            return random.choice(self.responses[offer_type])
        return None


class SessionMemory:
    """Tracks offers and context within current session only."""
    
    def __init__(self, greeting_manager: GreetingManager):
        self.greeting_manager = greeting_manager
        self.reset()
        
    def reset(self):
        """Clear all session memory (called on sleep)."""
        self.last_offer = None
        self.pending_offers = []
        self.context = {}
        
    def set_offer(self, greeting_data: Dict[str, Any]):
        """Set offers from greeting data structure."""
        self.last_offer = greeting_data['text']
        self.pending_offers = greeting_data.get('offers', [])
            
    def has_pending_offer(self) -> bool:
        """Check if there's a pending offer to respond to."""
        return len(self.pending_offers) > 0
        
    def get_offer_response(self, user_input: str) -> Optional[str]:
        """Generate response if user is responding to a remembered offer."""
        if not self.has_pending_offer():
            return None
            
        user_lower = user_input.lower()
        
        # Check for positive responses
        if any(word in user_lower for word in ["yes", "yeah", "sure", "okay", "ok", "please", "go ahead"]):
            # Use the first pending offer
            offer_type = self.pending_offers[0]
            response = self.greeting_manager.get_response_for_offer(offer_type)
            self.pending_offers = []  # Clear offers after responding
            return response or "Well, here I am! What did you have in mind?"
            
        # Check for negative responses
        if any(word in user_lower for word in ["no", "nah", "not now", "maybe later", "nope"]):
            response = "No worries! Maybe next time."
            self.pending_offers = []  # Clear offers
            return response
            
        return None


def _lmstudio_chat(messages: List[Dict[str, str]],
                   model: str = None,
                   api_base: str = None,
                   api_key: str = None,
                   temperature: float = 0.6,
                   max_tokens: int = 512) -> str:
    """Same LM Studio integration as original buddy.py"""
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
    """Same Groq transcription as original buddy.py"""
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
    }
    headers = {
        'Authorization': f'Bearer {api_key}',
    }
    try:
        r = requests.post(url, headers=headers, data=data, files=files, timeout=120)
        r.raise_for_status()
        j = r.json()
        txt = (j.get('text') or '').strip()
        return txt
    except Exception as e:
        LOG.warning(f'Groq STT failed: {e}')
        return ''


# Copy HeadMotion and STT classes from original buddy.py
try:
    import sys as _sys
    _sys.path.append('/home/pi/MasterPi')
    from common.ros_robot_controller_sdk import Board  # type: ignore
    _BOARD_OK = True
except Exception:
    Board = None  # type: ignore
    _BOARD_OK = False


class HeadMotion:
    """Minimal helper to nod and pitch head up/down (servo 3)."""
    def __init__(self):
        self.board = Board() if _BOARD_OK else None
        self.servo_id = int(os.environ.get('HEAD_PITCH_SERVO', '3'))
        self.HEAD_UP = int(os.environ.get('HEAD_UP', '650'))
        self.HEAD_DOWN = int(os.environ.get('HEAD_DOWN', '900'))
        if self.board:
            try:
                for ch in (1, 2, 3, 4, 5, 6):
                    try:
                        self.board.pwm_servo_enable(ch, True)
                    except Exception:
                        pass
            except Exception:
                pass

    def pitch(self, value: int, time_s: float = 0.08):
        if not self.board:
            return
        try:
            self.board.pwm_servo_set_position(time_s, [[self.servo_id, value]])
        except Exception:
            pass

    def look_up(self):
        self.pitch(self.HEAD_UP, 0.08)

    def look_down(self):
        self.pitch(self.HEAD_DOWN, 0.08)

    def nod(self):
        if not self.board:
            return
        seq = [self.HEAD_UP, self.HEAD_DOWN, self.HEAD_UP, int((self.HEAD_UP + self.HEAD_DOWN) / 2)]
        for v in seq:
            self.pitch(v, 0.08)
            time.sleep(0.15)


class STT:
    """Same STT class as original buddy.py - simplified for space"""
    def __init__(self, device_index: int | None = None, vosk_model_path: str | None = None, use_groq: Optional[bool] = None):
        self._recognizer = None
        self._mic = None
        self._device_index = device_index
        self._vosk_model = None
        if use_groq is None:
            self._use_groq = bool(os.environ.get('GROQ_API_KEY'))
        else:
            self._use_groq = bool(use_groq)
        
        # Initialize speech recognition - same as original
        try:
            import speech_recognition as sr  # type: ignore
            self._recognizer = sr.Recognizer()
            try:
                with _suppress_alsa():
                    self._mic = sr.Microphone(device_index=device_index, sample_rate=16000, chunk_size=1024)
                with self._mic as source:
                    self._recognizer.dynamic_energy_threshold = True
                    self._recognizer.energy_threshold = int(os.environ.get('BUDDY_ENERGY_THRESHOLD', '200'))
                    self._recognizer.pause_threshold = float(os.environ.get('BUDDY_PAUSE_THRESHOLD', '0.6'))
                    self._recognizer.non_speaking_duration = float(os.environ.get('BUDDY_NON_SPEAKING', '0.3'))
                    self._recognizer.adjust_for_ambient_noise(source, duration=0.8)
                LOG.info(f'🎙️  Microphone ready')
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
            return input('You: ').strip()
        import speech_recognition as sr  # type: ignore
        print(prompt, end='', flush=True)
        try:
            with self._mic as source:
                with _suppress_alsa():
                    audio = self._recognizer.listen(source, timeout=10, phrase_time_limit=12)
            
            # Try Groq first if available
            if self._use_groq:
                try:
                    wav = audio.get_wav_data(convert_rate=16000, convert_width=2)
                    txt = _groq_transcribe_wav(wav)
                    if txt:
                        return txt
                except Exception:
                    pass
            
            # Fallback to Google
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
    p = argparse.ArgumentParser(description='Bruno Buddy Enhanced - Wake-up System')
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

    # Initialize components
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

    # Initialize enhanced components
    greeting_manager = GreetingManager()
    session_memory = SessionMemory(greeting_manager)
    head = HeadMotion()
    
    # Start in sleep mode
    print("🤖 Bruno Enhanced Buddy starting in sleep mode...")
    head.look_down()
    time.sleep(1)

    messages: List[Dict[str, str]] = [{'role': 'system', 'content': args.system}]
    wake_info = f" Say '{args.wake}' to wake me; say 'go to sleep' to return to sleep." if args.wake else ''
    print(f"😴 Bruno is sleeping.{wake_info} Say 'stop' to exit.\n")
    
    LOG.info(f"Loaded {len(greeting_manager.greetings)} greetings from assets")

    try:
        empty_in_a_row = 0
        woken = False
        first_wake_this_session = True
        
        while True:
            user_text = stt.listen_once()
            if not user_text:
                empty_in_a_row += 1
                if empty_in_a_row >= 3 and not stt.available:
                    continue
                if empty_in_a_row >= 3 and stt.available:
                    typed = input('You (type, fallback): ').strip()
                    if typed:
                        user_text = typed
                        empty_in_a_row = 0
                    else:
                        continue
                continue
            empty_in_a_row = 0
            
            # Global stop intent
            def _said_stop(t: str) -> bool:
                return bool(re.search(r"\b(stop|quit|exit)\b", t.lower()))

            if _said_stop(user_text):
                print('👋 Bye!')
                head.look_down()
                break

            # Interrupt speaking
            if tts and re.search(r"\b(shut up|be quiet|stop speaking)\b", user_text.lower()):
                try:
                    tts.interrupt()
                    print('(stopped speaking)')
                except Exception:
                    pass
                continue

            # Enhanced wake/sleep logic
            if args.wake:
                low = user_text.lower()
                wake = args.wake.lower()
                
                if not woken:
                    # Sleeping - only respond to wake phrase
                    if wake not in low:
                        continue
                    
                    # WAKE UP SEQUENCE
                    idx = low.find(wake)
                    user_text = user_text[:idx] + user_text[idx+len(args.wake):]
                    user_text = user_text.strip() or 'hello'
                    woken = True
                    first_wake_this_session = True
                    print("(🌅 waking up)")
                    
                    # Physical wake up
                    try:
                        head.look_up()
                        time.sleep(0.3)
                        head.nod()
                    except Exception:
                        pass
                    
                    # DYNAMIC GREETING (only on first wake)
                    greeting_data = greeting_manager.get_random_greeting()
                    greeting_text = greeting_data['text']
                    session_memory.set_offer(greeting_data)
                    
                    print(f'Bruno: {greeting_text}\n')
                    if tts:
                        try:
                            tts.speak_sync(greeting_text)
                        except Exception:
                            pass
                    
                    first_wake_this_session = False
                    continue
                    
                else:
                    # Already awake
                    # Handle sleep intents
                    if any(p in low for p in ("go to sleep", "sleep now", "sleep bruno")):
                        woken = False
                        first_wake_this_session = True
                        session_memory.reset()  # Clear session memory
                        print(f"😴 (sleeping — say '{args.wake}' to wake me)")
                        try:
                            head.look_down()
                        except Exception:
                            pass
                        continue
                    
                    # If user repeats wake phrase, strip it but stay awake
                    if wake in low:
                        idx = low.find(wake)
                        user_text = (user_text[:idx] + user_text[idx+len(args.wake):]).strip() or 'hello'

            print(f'You: {user_text}')
            
            # Check for follow-up responses to remembered offers
            offer_response = session_memory.get_offer_response(user_text)
            if offer_response:
                print(f'Bruno: {offer_response}\n')
                if tts:
                    try:
                        tts.speak_sync(offer_response)
                    except Exception:
                        pass
                continue

            # Regular conversation with LM Studio
            messages.append({'role': 'user', 'content': user_text})
            try:
                reply = _lmstudio_chat(messages)
            except Exception as e:
                reply = f"[LM Studio error] {e}"
            messages.append({'role': 'assistant', 'content': reply})

            print(f'Bruno: {reply}\n')
            if tts:
                try:
                    tts.speak_sync(reply)
                except Exception:
                    pass
                    
    except KeyboardInterrupt:
        print('\n👋 Bye!')
    finally:
        try:
            if tts:
                tts.stop(wait=False)
        except Exception:
            pass
        try:
            head.look_down()
        except Exception:
            pass


if __name__ == '__main__':
    main()