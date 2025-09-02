# Bruno External Camera Surveillance (Refactor)

Multi‑file refactor of the external camera → local captioner → LM Studio summary pipeline with ultrasonic safety and emergency logic.

## Quick start

```bash
cd bruno_surveillance
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
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
```

## Notes
- `captioner.py` is a minimal stub that tries HF `transformers` if present. Replace it with your existing local captioner for best results.
- Hiwonder SDK modules (e.g., `common.sonar`, `common.mecanum`) come from your robot image under `/home/pi/MasterPi`.
- The optional `vision_obstacles.py` helper is not wired by default; integrate where desired in `app.py`.
