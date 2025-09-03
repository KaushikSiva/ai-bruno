#!/usr/bin/env python3
import os, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from utils import paths
from audio_tts import TTSSpeaker


def main():
    # Read env
    url = os.environ.get('INWORLD_TTS_URL')
    key = os.environ.get('INWORLD_API_KEY')
    voice = os.environ.get('BRUNO_AUDIO_VOICE', 'Ashley')
    if not url or not key:
        print('Set INWORLD_TTS_URL and INWORLD_API_KEY before running this test.')
        sys.exit(1)

    sp = TTSSpeaker(enabled=True, voice=voice)
    # Call the synth function directly and write output
    audio = sp._synthesize_tts("Hey, I'm Bruno. This is a TTS test.")
    if not audio:
        print('TTS request failed. Check INWORLD_API_KEY and service access.')
        sys.exit(2)
    out = paths.debug / 'inworld_tts_test.wav'
    out.write_bytes(audio)
    print(f'Wrote {out}')

    # Play the audio using the same routine the app uses
    try:
        sp._play_audio(audio)
    except Exception as e:
        print(f'Playback failed: {e}. File saved at {out}')

if __name__ == '__main__':
    main()
