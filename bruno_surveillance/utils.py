import os, sys, logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
def setup_logging_to_stdout(name:str='bruno.dual_camera')->logging.Logger:
    logger=logging.getLogger(name); logger.propagate=False; logger.setLevel(logging.INFO)
    if logger.handlers: logger.handlers.clear()
    sh=logging.StreamHandler(stream=sys.stdout); sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')); logger.addHandler(sh)
    logs_dir=Path.cwd()/'logs'; logs_dir.mkdir(parents=True, exist_ok=True)
    fh=RotatingFileHandler(str(logs_dir/'bruno.log'),maxBytes=5_000_000,backupCount=5)
    fh.setLevel(logging.INFO); fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')); logger.addHandler(fh)
    logger.info('Logging initialized (stdout + ./logs/bruno.log)'); return logger
LOG=setup_logging_to_stdout('bruno.dual_camera')
@dataclass
class Paths:
    base: Path; gpt_images: Path; logs: Path; debug: Path
    def save_image_path(self,prefix:str)->Path:
        import time; ts=time.strftime('%Y%m%d_%H%M%S'); us=int((time.time()*1_000_000)%1_000_000)
        return self.gpt_images/f'{prefix}_{ts}_{us:06d}.jpg'
    def sidecar_txt_path(self,img_path:Path)->Path: return img_path.with_suffix('.txt')
    @staticmethod
    def safe_write_text(path:Path, content:str):
        try: path.write_text(content,encoding='utf-8'); LOG.info(f'💾 Saved: {path}')
        except Exception as e: LOG.warning(f'⚠️  Failed to write file {path}: {e}')
def ensure_relative_paths()->Paths:
    base=Path.cwd(); p=Paths(base=base,gpt_images=base/'gpt_images',logs=base/'logs',debug=base/'debug')
    for d in (p.gpt_images,p.logs,p.debug): d.mkdir(parents=True,exist_ok=True)
    LOG.info(f'Working directory: {p.base}'); LOG.info(f'Images → {p.gpt_images}'); LOG.info(f'Logs → {p.logs}'); return p
paths=ensure_relative_paths(); save_image_path=paths.save_image_path; sidecar_txt_path=paths.sidecar_txt_path; safe_write_text=paths.safe_write_text
def load_env():
    if os.path.exists('.env'):
        for line in Path('.env').read_text().splitlines():
            if line.strip() and not line.startswith('#'):
                k,v=line.strip().split('=',1); os.environ[k]=v
        LOG.info('✓ Loaded .env file')
def get_env_str(key:str, default:str)->str: return os.environ.get(key, default)
def get_env_int(key:str, default:int)->int:
    try: return int(os.environ.get(key,str(default)))
    except Exception: return default
def get_env_float(key:str, default:float)->float:
    try: return float(os.environ.get(key,str(default)))
    except Exception: return default
