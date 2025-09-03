from typing import Tuple
from utils import LOG
from camera_builtin import BuiltinCamera
from camera_external import ExternalCamera


class CameraBase:
    def open(self) -> bool: raise NotImplementedError
    def read(self) -> Tuple[bool, object]: raise NotImplementedError
    def get_fresh_frame(self, max_attempts: int = 3, settle_reads: int = 3): raise NotImplementedError
    def reopen(self) -> None: raise NotImplementedError
    def release(self) -> None: raise NotImplementedError


def make_camera(mode: str, retry_attempts: int, retry_delay: float):
    if mode == 'builtin':
        LOG.info('📷 Using BuiltinCamera strategy')
        return BuiltinCamera(retry_attempts=retry_attempts, retry_delay=retry_delay)
    LOG.info('📷 Using ExternalCamera strategy')
    return ExternalCamera(retry_attempts=retry_attempts, retry_delay=retry_delay)

