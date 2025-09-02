import os, time, threading, urllib.request
from typing import Optional, Tuple
from urllib.parse import urlparse

import cv2
from utils import LOG

def _is_stream_up(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False

def _looks_like_local(url: str, default_port: int = 8080) -> bool:
    try:
        u = urlparse(url)
        if u.scheme not in ('http','https'):
            return False
        host = (u.hostname or '').lower()
        port = u.port or (443 if u.scheme == 'https' else 80)
        return host in ('127.0.0.1','localhost') and port == default_port
    except Exception:
        return False

def _start_stream_background(host='0.0.0.0', port=8080):
    try:
        import live_stream_test
        LOG.info(f'Starting background stream on {host}:{port}')
    except Exception as e:
        LOG.error(f'Could not import live_stream_test for autostart: {e}')
        return None
    t = threading.Thread(target=live_stream_test.run_stream, kwargs={'host':host,'port':port}, daemon=True)
    t.start()
    return t

def _open_url(url: str) -> Tuple[Optional[cv2.VideoCapture], str]:
    methods = [
        ('Built-in stream', lambda: cv2.VideoCapture(url)),
        ('Built-in stream FFMPEG', lambda: cv2.VideoCapture(url, cv2.CAP_FFMPEG)),
    ]
    for name, fn in methods:
        try:
            LOG.info(f'   Trying {name}…')
            cap = fn()
            if cap and cap.isOpened():
                for _ in range(4):
                    cap.read(); time.sleep(0.02)
                ok, frame = cap.read()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    LOG.info(f'   ✅ {name} SUCCESS! Resolution: {w}x{h}')
                    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception: pass
                    return cap, name
                cap.release()
        except Exception as e:
            LOG.warning(f'   ❌ {name} error: {e}')
    return None, 'Failed'

class BuiltinCamera:
    def __init__(self, retry_attempts: int = 3, retry_delay: float = 2.0):
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.cap = None
        self.method = 'Unknown'
        self.url = os.environ.get('BRUNO_CAMERA_URL', 'http://127.0.0.1:8080?action=stream')
        self.port = 8080

    def open(self) -> bool:
        if _looks_like_local(self.url, self.port) and not _is_stream_up(self.url):
            LOG.info(f'Local stream {self.url} not running. Launching live_stream_test…')
            _start_stream_background(host='0.0.0.0', port=self.port)
            t0 = time.time()
            while time.time() - t0 < 8.0:
                if _is_stream_up(self.url): break
                time.sleep(0.5)

        for attempt in range(self.retry_attempts):
            LOG.info(f'📹 Built-in camera connection attempt {attempt+1}…')
            self.cap, self.method = _open_url(self.url)
            if self.cap:
                LOG.info(f'✅ Built-in camera connected via {self.method}')
                return True
            if attempt < self.retry_attempts - 1:
                LOG.info(f'⏳ Retrying in {self.retry_delay} seconds…')
                time.sleep(self.retry_delay)
        LOG.error('❌ Failed to connect built-in camera')
        return False

    def is_open(self) -> bool:
        return bool(self.cap and self.cap.isOpened())

    def read(self) -> Tuple[bool, object]:
        if not self.is_open(): return False, None
        ok, frame = self.cap.read()
        return (ok and frame is not None), (frame if ok else None)

    def get_fresh_frame(self, max_attempts: int = 3, settle_reads: int = 3):
        if not self.is_open(): return None
        try:
            for _ in range(settle_reads):
                self.cap.read(); time.sleep(0.02)
            for _ in range(max_attempts):
                ok, f = self.cap.read()
                if ok and f is not None: return f
                time.sleep(0.03)
        except Exception:
            return None
        return None

    def reopen(self):
        try: self.release()
        except Exception: pass
        time.sleep(1.0); self.open()

    def release(self):
        try:
            if self.cap: self.cap.release()
        except Exception: pass
        self.cap = None; self.method = 'Unknown'
