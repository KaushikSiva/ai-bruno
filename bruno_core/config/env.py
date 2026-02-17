import os
from pathlib import Path

ENV_DEFAULTS = {
    "CAM_MODE": "external",
    "BRUNO_AUDIO_VOICE": "Dominoux",
    "BRUNO_AUDIO_ENABLED": "0",
    "CAPTION_BACKEND": "local",
    "BUDDY_WAKE": "hey bruno",
    "VOSK_MODEL": "",
    "HEAD_PITCH_SERVO": "3",
    "HEAD_UP": "650",
    "HEAD_DOWN": "900",
    "BUDDY_ENERGY_THRESHOLD": "150",
    "BUDDY_PAUSE_THRESHOLD": "1.0",
    "BUDDY_NON_SPEAKING": "0.8",
    "LLM_API_BASE": "http://localhost:1234/v1",
    "LLM_MODEL": "lmstudio",
    "LLM_API_KEY": "lm-studio",
    "LLM_TIMEOUT_SEC": "30",
    "GROQ_API_BASE": "https://api.groq.com/openai/v1",
    "GROQ_API_KEY": "",
    "GROQ_STT_MODEL": "whisper-large-v3",
    "GROQ_VISION_MODEL": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "INWORLD_TTS_URL": "https://api.inworld.ai/tts/v1/voice:stream",
    "INWORLD_API_KEY": "",
    "PHOTO_INTERVAL_SEC": "15",
    "SUMMARY_DELAY_SEC": "120",
    "ULTRA_CAUTION_CM": "50.0",
    "ULTRA_DANGER_CM": "25.0",
    "BRUNO_SPEED": "40",
    "BRUNO_TURN_SPEED": "40",
    "BRUNO_TURN_TIME": "0.5",
    "BRUNO_BACKUP_TIME": "0.0",
    "BRUNO_CAMERA_URL": "http://127.0.0.1:8080?action=stream",
    "EXTERNAL_CAM_NO_ARM": "1",
    "BRUNO_ROVER_FORWARD_SPEED": "35",
    "BRUNO_ROVER_TURN_SPEED": "35",
    "BRUNO_ROVER_TURN_ROT": "0.45",
    "BRUNO_ROVER_MIN_MOTION_AREA": "1800",
    "BRUNO_ROVER_CENTER_DEADBAND": "0.20",
    "BRUNO_ROVER_RISK_STOP": "0.80",
    "BRUNO_ROVER_RISK_CAUTION": "0.60",
    "BRUNO_ROVER_CONFIDENCE_MIN": "0.45",
    "BRUNO_ROVER_STOP_HOLD_S": "0.8",
    "BRUNO_ROVER_UNCERTAIN_FRAME_LIMIT": "10",
    "BRUNO_ROVER_SEARCH_SWITCH_S": "1.5",
    "BRUNO_USE_VLM": "1",
    "BRUNO_VLM_PROVIDER": "auto",
    "BRUNO_VLM_LOCAL_BASE": "http://127.0.0.1:8080/v1",
    "BRUNO_VLM_REMOTE_BASE": "http://localhost:1234/v1",
    "BRUNO_VLM_API_BASE": "http://localhost:1234/v1",
    "BRUNO_VLM_LOCAL_MODEL": "gemma3",
    "BRUNO_VLM_REMOTE_MODEL": "gemma3",
    "BRUNO_VLM_MODEL": "gemma3",
    "BRUNO_VLM_API_KEY": "lm-studio",
    "BRUNO_VLM_LOCAL_TIMEOUT_MS": "800",
    "BRUNO_VLM_REMOTE_TIMEOUT_MS": "1600",
    "BRUNO_VLM_MAX_LATENCY_MS": "1200",
    "BRUNO_VLM_MIN_CONFIDENCE": "0.45",
    "BRUNO_VLM_INTERVAL_SEC": "1.5",
    "BRUNO_VLM_COOLDOWN_S": "5.0",
    "STREAM_SOURCE": "0",
    "HOST": "0.0.0.0",
    "PORT": "8080",
}


def load_env(env_path: str | Path = ".env") -> None:
    p = Path(env_path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ[key] = val


def get_env_str(key: str, default: str | None = None) -> str:
    fallback = ENV_DEFAULTS.get(key, "" if default is None else default)
    return os.environ.get(key, fallback if default is None else default)


def get_env_int(key: str, default: int | None = None) -> int:
    if default is None:
        default = int(ENV_DEFAULTS.get(key, "0"))
    try:
        return int(os.environ.get(key, str(default)))
    except Exception:
        return default


def get_env_float(key: str, default: float | None = None) -> float:
    if default is None:
        default = float(ENV_DEFAULTS.get(key, "0"))
    try:
        return float(os.environ.get(key, str(default)))
    except Exception:
        return default


def get_env_bool(key: str, default: bool | None = None) -> bool:
    if default is None:
        default = ENV_DEFAULTS.get(key, "0").strip().lower() in ("1", "true", "yes", "on")
    return os.environ.get(key, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")
