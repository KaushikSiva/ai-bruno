from bruno_core.config.env import get_env_str
from bruno_core.inference.providers.groq_vision import get_caption as groq_get_caption
from bruno_core.logging.setup import LOG


_BACKEND = "local"


def set_backend(name: str = "local") -> None:
    global _BACKEND
    name = (name or "local").strip().lower()
    if name not in ("local", "groq"):
        LOG.warning("Unknown caption backend %s; using local", name)
        name = "local"
    _BACKEND = name


def caption_image(image_path: str) -> str:
    if _BACKEND == "groq":
        try:
            return groq_get_caption(image_path)
        except Exception as exc:
            LOG.warning("Groq caption failed: %s", exc)
            return "[caption failed]"
    return "[local caption backend not configured]"


set_backend(get_env_str("CAPTION_BACKEND"))
