#!/usr/bin/env python3
# coding: utf-8
"""
Minimal entrypoint. All logic lives in controller.py
"""
import os
import argparse
from controller import run as run_controller


def main():
    parser = argparse.ArgumentParser(description='Bruno Dual-Mode Surveillance')
    parser.add_argument('--mode', choices=['builtin', 'external'], default=os.environ.get('CAM_MODE', 'external'), help='Camera mode')
    parser.add_argument('--audio', '--talk', dest='audio', action='store_true', help='Enable TTS speech')
    parser.add_argument('--voice', default=os.environ.get('BRUNO_AUDIO_VOICE', 'alloy'), help='TTS voice')
    args = parser.parse_args()

    audio_env = os.environ.get('BRUNO_AUDIO_ENABLED', '').strip().lower() in ('1','true','yes','on')
    audio_enabled = args.audio or audio_env

    run_controller(mode=args.mode, audio_enabled=audio_enabled, audio_voice=args.voice)


if __name__ == '__main__':
    main()
