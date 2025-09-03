#!/usr/bin/env python3
# coding: utf-8
"""
Minimal entrypoint. All logic lives in controller.py
"""
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Bruno Dual-Mode Surveillance')
    parser.add_argument('--mode', choices=['builtin', 'external'], default=os.environ.get('CAM_MODE', 'external'), help='Camera mode')
    parser.add_argument('--audio', '--talk', dest='audio', action='store_true', help='Enable TTS speech')
    parser.add_argument('--voice', default=os.environ.get('BRUNO_AUDIO_VOICE', 'alloy'), help='TTS voice')
    parser.add_argument('--caption-backend', choices=['local','groq'], default=os.environ.get('CAPTION_BACKEND','local'), help='Choose captioning backend')
    args = parser.parse_args()

    audio_env = os.environ.get('BRUNO_AUDIO_ENABLED', '').strip().lower() in ('1','true','yes','on')
    audio_enabled = args.audio or audio_env

    # Apply caption backend selection before controller imports/use
    try:
        from caption_pipeline import set_backend as _set_caption_backend
        _set_caption_backend(args.caption_backend)
    except Exception:
        pass

    # Import controller lazily so caption backend is set first
    from controller import run as run_controller
    run_controller(mode=args.mode, audio_enabled=audio_enabled, audio_voice=args.voice)


if __name__ == '__main__':
    main()
