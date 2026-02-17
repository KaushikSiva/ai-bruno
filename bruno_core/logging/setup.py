import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def init_logging(app_name: str = "bruno", log_dir: str | Path = "logs", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(app_name)
    logger.propagate = False
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s")
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(str(log_path / f"{app_name}.log"), maxBytes=5_000_000, backupCount=5)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


LOG = init_logging("bruno")

