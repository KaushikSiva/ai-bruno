import time
from dataclasses import dataclass
from pathlib import Path

from bruno_core.logging.setup import LOG


@dataclass
class Paths:
    base: Path
    gpt_images: Path
    logs: Path
    debug: Path

    def save_image_path(self, prefix: str) -> Path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        us = int((time.time() * 1_000_000) % 1_000_000)
        return self.gpt_images / f"{prefix}_{ts}_{us:06d}.jpg"

    def sidecar_txt_path(self, img_path: Path) -> Path:
        return img_path.with_suffix(".txt")

    @staticmethod
    def safe_write_text(path: Path, content: str) -> None:
        try:
            path.write_text(content, encoding="utf-8")
            LOG.info("Saved: %s", path)
        except Exception as exc:
            LOG.warning("Failed to write file %s: %s", path, exc)


def ensure_relative_paths(base: Path | None = None) -> Paths:
    root = base or Path.cwd()
    p = Paths(base=root, gpt_images=root / "gpt_images", logs=root / "logs", debug=root / "debug")
    for d in (p.gpt_images, p.logs, p.debug):
        d.mkdir(parents=True, exist_ok=True)
    return p


paths = ensure_relative_paths()

