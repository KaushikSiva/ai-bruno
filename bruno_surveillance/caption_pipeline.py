import os
from utils import LOG

_BACKEND = 'local'
_local = None
_groq = None


def set_backend(name: str = 'local') -> None:
    """Select caption backend at runtime: 'local' (HF) or 'groq'."""
    global _BACKEND, _local, _groq
    name = (name or 'local').strip().lower()
    if name not in ('local', 'groq'):
        LOG.warning(f'Unknown caption backend {name}; falling back to local')
        name = 'local'
    _BACKEND = name
    if name == 'groq':
        _local = None
        try:
            import groq_vision as _g
            _groq = _g
            LOG.info('Caption backend: Groq Vision')
        except Exception as _e:
            LOG.warning(f'Groq vision backend import failed: {_e}')
            _groq = None
    else:
        _groq = None
        try:
            import captioner as _l
            _local = _l
            LOG.info('Caption backend: Local (HF pipeline)')
        except Exception as _e:
            _local = None


def caption_image(image_path: str) -> str:
    if _BACKEND == 'groq':
        if _groq is None:
            return '[groq vision backend not available]'
        try:
            return _groq.get_caption(image_path)
        except Exception as e:
            LOG.warning(f'Groq caption failed: {e}')
            return '[caption failed]'

    # default local
    if _local is None:
        return '[captioner module not available]'
    try:
        return _local.get_caption(image_path)
    except Exception as e:
        LOG.warning(f'Captioner failed: {e}')
        return '[caption failed]'


# Initialize from environment on import
set_backend(os.environ.get('CAPTION_BACKEND', 'local'))
