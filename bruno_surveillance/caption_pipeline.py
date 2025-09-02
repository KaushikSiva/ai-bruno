from utils import LOG
try:
    import captioner  # your existing local module
except Exception as _e:
    captioner = None

def caption_image(image_path: str) -> str:
    if captioner is None:
        return "[captioner module not available]"
    try:
        return captioner.get_caption(image_path)
    except Exception as e:
        LOG.warning(f'Captioner failed: {e}')
        return '[caption failed]'
