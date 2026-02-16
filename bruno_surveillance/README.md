# Bruno Dual-Mode Surveillance

Switch between **builtin** and **external** camera without changing code.

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

# Optional: Gemini robotics policy (AI-driven motion)
ROBOT_POLICY=                 # set to 'gemini' to enable
GEMINI_API_KEY=               # or GOOGLE_API_KEY
GEMINI_MODEL=gemini-robotics-er-1.5-preview  # recommended robotics model
GEMINI_API_BASE=https://generativelanguage.googleapis.com
GEMINI_USE_SDK=1              # use google-genai SDK if installed
GEMINI_TARGET_OBJECT=water bottle  # optional: goal-directed approach behavior
GEMINI_TARGET_MODE=bbox       # 'bbox' for deterministic centering, or 'policy' (default)
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

## Gemini Robotics Policy (optional)
- Enable a lightweight robotics policy powered by Gemini to propose short, safe motions after each snapshot.
  - Set `ROBOT_POLICY=gemini` and provide `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).
  - Recommended model: `gemini-robotics-er-1.5-preview`.
  - To use Google’s official SDK: `pip install google-genai` and set `GEMINI_USE_SDK=1`.
- The controller sends the latest image + caption and expects compact JSON: `{action, duration_s, speed, reason}`.
- Actions map to Bruno primitives: `forward`, `reverse`, `left`, `right`, `stop`. Duration is clamped to 0..2s.
- Safety overrides remain active via ultrasonic checks.

### Approach a target (e.g., water bottle)
- Set `GEMINI_TARGET_OBJECT="water bottle"` to bias behavior toward approaching that object.
- Choose mode:
  - `GEMINI_TARGET_MODE=bbox` for deterministic behavior (detect bbox, center, then approach).
  - `GEMINI_TARGET_MODE=policy` (default) to let the LLM propose actions directly.
- Behavior:
  - If the target isn’t visible, the policy suggests a short scan (left/right).
  - If the target is visible but off-center, it suggests a small turn to center it.
  - If centered and safe (per ultrasonic), it suggests a short forward move.

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
