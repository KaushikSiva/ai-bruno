#!/usr/bin/env python3
"""
Raspberry Pi Audio Debug Script
This script helps debug audio issues on Pi by testing different configurations.
"""

import os
import subprocess
import sys
from contextlib import contextmanager

@contextmanager  
def _suppress_alsa():
    """Raspberry Pi optimized ALSA/JACK/Audio error suppression."""
    try:
        old_env = {}
        audio_env_vars = {
            'ALSA_PCM_CARD': '0',
            'ALSA_PCM_DEVICE': '0', 
            'ALSA_LOG_LEVEL': '0',
            'ALSA_PLUGIN_DIR': '/usr/lib/arm-linux-gnueabihf/alsa-lib',
            'ALSA_MIXER_SIMPLE': '1',
            'JACK_NO_START_SERVER': '1',
            'JACK_NO_AUDIO_RESERVATION': '1', 
            'JACK_SILENCE_MESSAGES': '1',
            'JACK_DEFAULT_SERVER': 'dummy',
            'JACK_DRIVER': 'dummy',
            'PULSE_LATENCY_MSEC': '30',
            'PA_ALSA_PLUGHW': '1',
            'OSS_AUDIODEV': '/dev/null',
            'OSS_MIXERDEV': '/dev/null',
            'AUDIODEV': '/dev/null',
            'AUDIODRIVER': 'pulse',
            'SDL_AUDIODRIVER': 'pulse',
            'LIBASOUND_DEBUG': '0',
            'ALSA_PERIOD_TIME': '0',
            'ALSA_BUFFER_TIME': '0',
            'ALSA_PCM_STREAM': '0',
            'ALSA_RAWMIDI_STREAM': '0'
        }
        
        for key, value in audio_env_vars.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        
        yield
        
    finally:
        try:
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            
            for key, old_value in old_env.items():
                if old_value is not None:
                    os.environ[key] = old_value
                elif key in os.environ:
                    del os.environ[key]
        except Exception:
            pass

def check_audio_system():
    """Check what audio systems are available."""
    print("=== Audio System Check ===")
    
    # Check PulseAudio
    try:
        result = subprocess.run(['pulseaudio', '--check'], capture_output=True)
        if result.returncode == 0:
            print("✓ PulseAudio is running")
        else:
            print("⚠ PulseAudio is not running")
    except FileNotFoundError:
        print("✗ PulseAudio not installed")
    
    # Check ALSA
    try:
        result = subprocess.run(['aplay', '-l'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ ALSA devices available:")
            print(result.stdout)
        else:
            print("⚠ ALSA devices check failed")
    except FileNotFoundError:
        print("✗ ALSA not available")
    
    # Check audio players
    players = ['paplay', 'aplay', 'ffplay']
    for player in players:
        try:
            result = subprocess.run([player, '--version'], capture_output=True)
            print(f"✓ {player} available")
        except FileNotFoundError:
            print(f"✗ {player} not found")

def test_microphone():
    """Test microphone with different configurations."""
    print("\n=== Microphone Test ===")
    
    try:
        import speech_recognition as sr
        print("✓ SpeechRecognition imported")
        
        # Test with suppression
        with _suppress_alsa():
            try:
                mics = sr.Microphone.list_microphone_names()
                print(f"✓ Found {len(mics)} microphone(s):")
                for i, mic in enumerate(mics):
                    print(f"  [{i}] {mic}")
                
                # Test different device configurations
                test_configs = [
                    (None, "Auto-detect"),
                    (0, "Device 0"),
                ]
                
                for device_idx, desc in test_configs:
                    try:
                        with _suppress_alsa():
                            mic = sr.Microphone(device_index=device_idx, sample_rate=16000, chunk_size=1024)
                        print(f"✓ {desc}: Microphone initialized successfully")
                    except Exception as e:
                        print(f"✗ {desc}: {e}")
                        
            except Exception as e:
                print(f"✗ Microphone enumeration failed: {e}")
                
    except ImportError:
        print("✗ SpeechRecognition not installed")
        print("  Run: pip install SpeechRecognition pyaudio")

if __name__ == '__main__':
    print("Pi Audio Debug Script")
    print("=====================")
    check_audio_system()
    test_microphone()
    print("\nTo fix issues:")
    print("1. For PulseAudio: systemctl --user start pulseaudio")
    print("2. For ALSA: Check 'alsamixer' and 'speaker-test'") 
    print("3. For permissions: Add user to 'audio' group")