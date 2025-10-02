# Bruno the Buddy

youtube video : https://www.youtube.com/watch?v=d_K3juzPHkk

 Meet Bruno, a smart robot powered by GPT-OSS that combines surveillance, obstacle avoidance, and face-tracking with a playful side. He chats, teaches math and science, and even follows you like a buddy. Today, he’s a guardian and learning partner. Tomorrow, he’s a sustainability hero, picking up plastic bottles and helping the environment. Bruno isn’t just a robot; he’s a platform for security, education, and innovation in your home.

## Tech Stack

Language: Python 3.10+.
Computer Vision: OpenCV for image capture, tracking, and preprocessing.
Image-to-Text:  Maverick for scene captioning.
AI Models: GPT-OSS, Groq(Meta Maverick)
Frameworks: Flask for API endpoints and control server.
Hiwonder SDK: Python library to control motors, sensors, and servos.
Speech: SpeechRecognition + Whisper (via Groq API) for STT; TTSSpeaker
(Inworld/local) for TTS.

## Run
```bash
cd bruno_surveillance
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Using env
export CAM_MODE=builtin   # or external
python app.py

# Or CLI
python app.py --mode external
```

## Environment variables (`.env`)

```dotenv
PHOTO_INTERVAL_SEC=15
SUMMARY_DELAY_SEC=120

# LM Studio (OpenAI‑compatible)
LLM_API_BASE=http://localhost:1234/v1
LLM_MODEL=lmstudio
LLM_TIMEOUT_SEC=30
LLM_API_KEY=lm-studio

# Ultrasonic thresholds & motion
ULTRA_CAUTION_CM=50
ULTRA_DANGER_CM=25
BRUNO_SPEED=40
BRUNO_TURN_SPEED=40
BRUNO_TURN_TIME=0.5
BRUNO_BACKUP_TIME=0.0

# Audio (optional; off by default). Also supports CLI flags --audio / --talk
BRUNO_AUDIO_ENABLED=false
BRUNO_AUDIO_VOICE=default
BRUNO_AUDIO_BACKEND=inworld   # or local (uses tts_voice.speak)
BRUNO_AUDIO_LOCAL_MODULE=tts_voice.speak
BRUNO_AUDIO_LOCAL_FUNC=speak

# Inworld TTS endpoint (used when BRUNO_AUDIO_BACKEND=inworld)
INWORLD_TTS_URL=              # e.g., https://your-inworld-host/tts
INWORLD_API_KEY=              # optional; sent as Bearer token

# Caption backend (local HF or Groq Vision)
CAPTION_BACKEND=local        # or groq
GROQ_API_KEY=                # required if CAPTION_BACKEND=groq
GROQ_VISION_MODEL=llama-4-maverick-17b-128e-instruct
GROQ_API_BASE=https://api.groq.com/openai/v1
```

## Notes
- app.py is a minimal entrypoint; the main loop lives in `controller.py`.
- Cameras are modular: `camera_shared.py` chooses `camera_builtin.py` or `camera_external.py`.
- `captioner.py` is a minimal stub that tries HF `transformers` if present. Replace it with your existing local captioner for best results.
- Hiwonder SDK modules (e.g., `common.sonar`, `common.mecanum`) come from your robot image under `/home/pi/MasterPi`.

## Groq Vision backend
- To switch from local BLIP to Groq’s Llama‑4 Maverick vision model:
  - Set `CAPTION_BACKEND=groq` and `GROQ_API_KEY` in your environment or `.env`.
  - Optional: change `GROQ_VISION_MODEL` (defaults to `llama-4-maverick-17b-128e-instruct`).
  - The app will send each snapshot as a data URL to Groq’s OpenAI‑compatible `chat/completions` and use the response as the caption.
  - Or use CLI: `--caption-backend groq` (requires `GROQ_API_KEY`).

## Audio TTS (optional)
- When enabled (via `.env BRUNO_AUDIO_ENABLED=1` or `--audio`/`--talk`), Bruno speaks on a background thread:
  - On start: "Hey, I'm Bruno"
  - Every snapshot caption
  - Final LM Studio summary
- Uses Inworld TTS via a simple HTTP endpoint you configure (set `INWORLD_TTS_URL`, optional `INWORLD_API_KEY`).
- Playback uses `simpleaudio` if available, or falls back to `aplay`/`ffplay`.
- To use your local TTS (`tts_voice/speak.py`): set `--audio-backend local` (or `BRUNO_AUDIO_BACKEND=local`).
   - By default, it imports `tts_voice.speak` and calls the `speak(text)` function.
   - Override module/function via `BRUNO_AUDIO_LOCAL_MODULE` and `BRUNO_AUDIO_LOCAL_FUNC` if needed.

Env:
- `CAM_MODE` = builtin | external
- `--caption-backend` = local | groq (CLI override for CAPTION_BACKEND)
- `PHOTO_INTERVAL_SEC` (default 15)
- `SUMMARY_DELAY_SEC` (default 120)
- `LLM_API_BASE` (default http://localhost:1234/v1)
- `LLM_MODEL` (default lmstudio)
- Builtin camera only: `BRUNO_CAMERA_URL` (default http://127.0.0.1:8080?action=stream)
