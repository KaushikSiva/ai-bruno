#!/usr/bin/env python3
# coding: utf-8
"""
Minimal entrypoint. Runs surveillance by default; --buddy runs voice buddy.
"""
import os
import sys
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description='Bruno Dual-Mode Surveillance')
    parser.add_argument('--mode', choices=['builtin', 'external'], default=os.environ.get('CAM_MODE', 'external'), help='Camera mode')
    parser.add_argument('--audio', '--talk', dest='audio', action='store_true', help='Enable TTS speech')
    parser.add_argument('--voice', default=os.environ.get('BRUNO_AUDIO_VOICE', 'Ashley'), help='TTS voice/id')
    parser.add_argument('--caption-backend', choices=['local','groq'], default=os.environ.get('CAPTION_BACKEND','local'), help='Choose captioning backend')
    parser.add_argument('--buddy', action='store_true', help='Run conversational Buddy instead of surveillance')
    # Optional passthroughs for buddy (ignored by surveillance)
    parser.add_argument('--mic-index', type=int, default=None, help='Buddy mic device index')
    parser.add_argument('--wake', default=os.environ.get('BUDDY_WAKE', None), help='Buddy wake phrase (e.g., "hey bruno")')
    args = parser.parse_args()

    audio_env = os.environ.get('BRUNO_AUDIO_ENABLED', '').strip().lower() in ('1','true','yes','on')
    audio_enabled = args.audio or audio_env

    if args.buddy:
        # Run buddy.py as a subprocess to avoid argparse conflicts
        buddy_path = os.path.join(os.path.dirname(__file__), 'buddy.py')
        cmd = [sys.executable, buddy_path]
        if audio_enabled:
            cmd.append('--audio')
        if args.voice:
            cmd += ['--voice', args.voice]
        if args.mic_index is not None:
            cmd += ['--mic-index', str(args.mic_index)]
        if args.wake:
            cmd += ['--wake', args.wake]
        # Pass through Vosk model if set via env
        if os.environ.get('VOSK_MODEL'):
            cmd += ['--vosk-model', os.environ['VOSK_MODEL']]
        subprocess.call(cmd)
        return

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
