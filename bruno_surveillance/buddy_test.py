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

# Load environment variables from .env file
try:
    from pathlib import Path
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key and not key in os.environ:  # Don't override existing env vars
                        os.environ[key] = value
except Exception:
    pass  # Silently continue if .env loading fails

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
        """Generate intelligent response if user is responding to a remembered offer."""
        if not self.has_pending_offer():
            return None
            
        user_lower = user_input.lower()
        offer_type = self.pending_offers[0]
        
        # Check for negative responses first
        if any(word in user_lower for word in ["no", "nah", "not now", "maybe later", "nope"]):
            response = "No worries! Maybe next time."
            self.pending_offers = []  # Clear offers
            return response
        
        # Check for positive responses
        has_positive = any(word in user_lower for word in ["yes", "yeah", "sure", "okay", "ok", "please", "go ahead"])
        
        if has_positive or self._is_specific_request(user_input, offer_type):
            # Handle specific requests intelligently
            specific_response = self._handle_specific_offer_request(user_input, offer_type)
            if specific_response:
                self.pending_offers = []  # Clear offers after responding
                return specific_response
            
            # Fallback to generic response
            response = self.greeting_manager.get_response_for_offer(offer_type)
            self.pending_offers = []  # Clear offers after responding
            return response or "Well, here I am! What did you have in mind?"
            
        return None
    
    def _is_specific_request(self, user_input: str, offer_type: str) -> bool:
        """Check if user input contains specific requests related to the offer."""
        user_lower = user_input.lower()
        
        # Language-specific keywords
        if offer_type == "languages":
            language_keywords = [
                "japanese", "spanish", "french", "german", "italian", "chinese", "korean", 
                "russian", "portuguese", "arabic", "hindi", "dutch", "swedish", "norwegian",
                "japanese", "mandarin", "cantonese", "hebrew", "polish", "thai", "vietnamese"
            ]
            return any(lang in user_lower for lang in language_keywords)
        
        # Add more specific detections for other offer types
        elif offer_type == "joke":
            return any(word in user_lower for word in ["funny", "hilarious", "dad", "pun", "knock"])
        elif offer_type == "fun_fact":
            return any(word in user_lower for word in ["animal", "space", "science", "history", "nature"])
        elif offer_type == "dance":
            return any(word in user_lower for word in ["robot", "breakdance", "tango", "waltz", "hip"])
        
        return False
    
    def _handle_specific_offer_request(self, user_input: str, offer_type: str) -> Optional[str]:
        """Handle specific requests within offer responses."""
        user_lower = user_input.lower()
        
        if offer_type == "languages":
            # Map language requests to actual translations of "Hello"
            language_map = {
                "japanese": "こんにちは (Konnichiwa) - That's hello in Japanese!",
                "spanish": "¡Hola! - That's hello in Spanish!", 
                "french": "Bonjour! - That's hello in French!",
                "german": "Guten Tag! - That's hello in German!",
                "italian": "Ciao! - That's hello in Italian!",
                "chinese": "你好 (Nǐ hǎo) - That's hello in Chinese!",
                "korean": "안녕하세요 (Annyeonghaseyo) - That's hello in Korean!",
                "russian": "Здравствуйте (Zdravstvuyte) - That's hello in Russian!",
                "portuguese": "Olá! - That's hello in Portuguese!",
                "arabic": "مرحبا (Marhaba) - That's hello in Arabic!",
                "hindi": "नमस्ते (Namaste) - That's hello in Hindi!",
                "dutch": "Hallo! - That's hello in Dutch!",
                "swedish": "Hej! - That's hello in Swedish!",
                "hebrew": "שלום (Shalom) - That's hello in Hebrew!"
            }
            
            for language, translation in language_map.items():
                if language in user_lower:
                    return translation
        
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
        
        # Initialize speech recognition with Pi-optimized ALSA suppression
        try:
            import speech_recognition as sr  # type: ignore
            self._recognizer = sr.Recognizer()
            
            # Try multiple audio backends and device indices for Pi
            mic_init_attempts = []
            
            # Try user-specified device first if provided
            if device_index is not None:
                mic_init_attempts.append(lambda: sr.Microphone(device_index=device_index, sample_rate=16000, chunk_size=1024))
            
            # Try PulseAudio default (usually works best on Pi)
            mic_init_attempts.append(lambda: sr.Microphone(sample_rate=16000, chunk_size=1024))  # Auto-detect
            
            # Try ALSA device 0 specifically
            mic_init_attempts.append(lambda: sr.Microphone(device_index=0, sample_rate=16000, chunk_size=1024))
            
            # Try with different sample rates for compatibility
            mic_init_attempts.append(lambda: sr.Microphone(device_index=0, sample_rate=44100, chunk_size=2048))
            mic_init_attempts.append(lambda: sr.Microphone(sample_rate=44100, chunk_size=2048))
            
            for attempt, mic_factory in enumerate(mic_init_attempts):
                try:
                    with _suppress_alsa():
                        self._mic = mic_factory()
                        # Test microphone with enhanced suppression
                        with _suppress_alsa():
                            with self._mic as source:
                                self._recognizer.dynamic_energy_threshold = True
                                self._recognizer.energy_threshold = int(os.environ.get('BUDDY_ENERGY_THRESHOLD', '200'))
                                self._recognizer.pause_threshold = float(os.environ.get('BUDDY_PAUSE_THRESHOLD', '0.6'))
                                self._recognizer.non_speaking_duration = float(os.environ.get('BUDDY_NON_SPEAKING', '0.3'))
                                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)  # Shorter for Pi
                    LOG.info(f'🎙️  Microphone ready (attempt {attempt + 1})')
                    break
                except Exception as e:
                    if attempt == len(mic_init_attempts) - 1:
                        LOG.warning(f'No microphone after {len(mic_init_attempts)} attempts: {e}')
                        self._recognizer = None
                        self._mic = None
                    continue
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
            # Apply comprehensive suppression to entire audio operation
            with _suppress_alsa():
                with self._mic as source:
                    # Additional suppression during actual listening
                    with _suppress_alsa():
                        audio = self._recognizer.listen(source, timeout=10, phrase_time_limit=12)
            
            # Try Groq first if available (with suppression)
            if self._use_groq:
                try:
                    with _suppress_alsa():
                        wav = audio.get_wav_data(convert_rate=16000, convert_width=2)
                    txt = _groq_transcribe_wav(wav)
                    if txt:
                        return txt
                except Exception:
                    pass
            
            # Fallback to Google (with suppression)
            try:
                with _suppress_alsa():
                    text = self._recognizer.recognize_google(audio)
                return text.strip()
            except Exception:
                try:
                    with _suppress_alsa():
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
            old_env[key] = _os.environ.get(key)
            _os.environ[key] = value
        
        # Redirect file descriptors
        devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
        saved_stdout_fd = _os.dup(1)
        saved_stderr_fd = _os.dup(2)
        
        # Redirect both stdout and stderr to suppress audio messages
        _os.dup2(devnull_fd, 1)
        _os.dup2(devnull_fd, 2)
        _os.close(devnull_fd)
        
        yield
        
    finally:
        try:
            # Restore file descriptors
            _os.dup2(saved_stdout_fd, 1)
            _os.dup2(saved_stderr_fd, 2)
            _os.close(saved_stdout_fd)
            _os.close(saved_stderr_fd)
            
            # Restore environment variables
            for key, old_value in old_env.items():
                if old_value is not None:
                    _os.environ[key] = old_value
                elif key in _os.environ:
                    del _os.environ[key]
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

            # Enhanced conversation with LM Studio (include context from pending offers)
            context_enhanced_text = user_text
            if session_memory.has_pending_offer():
                offer_type = session_memory.pending_offers[0]
                last_offer = session_memory.last_offer
                context_enhanced_text = f"[Context: I just offered '{offer_type}' by saying '{last_offer}'] {user_text}"
                # Clear the offer since we're handling it through LLM now
                session_memory.pending_offers = []
            
            messages.append({'role': 'user', 'content': context_enhanced_text})
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