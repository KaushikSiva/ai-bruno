#!/usr/bin/env python3
# coding: utf-8
"""
Capture-only script: take N photos at a fixed interval (no movement).
Optional upload to a Mac via scp.
"""
import argparse
import time
import os
import subprocess
from typing import Optional

import cv2

from utils import LOG, paths
from camera_external import ExternalCamera
from camera_shared import read_or_reconnect


def upload_file(path: str, dest: Optional[str], port: int, timeout_s: int, retries: int) -> bool:
    if not dest:
        return True
    cmd = [
        "scp",
        "-P",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={timeout_s}",
        path,
        dest,
    ]
    for attempt in range(retries + 1):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            LOG.info(f"📤 Uploaded to {dest}")
            return True
        err = (result.stderr or "").strip()
        LOG.warning(f"📤 Upload failed (attempt {attempt + 1}/{retries + 1}): {err}")
        time.sleep(0.5)
    return False


def upload_gdrive(path: str, remote_path: Optional[str], timeout_s: int, retries: int) -> bool:
    if not remote_path:
        return True
    if remote_path.endswith("/") or remote_path.endswith(":"):
        remote_path = remote_path + os.path.basename(path)
    cmd = [
        "rclone",
        "copyto",
        path,
        remote_path,
        "--timeout",
        f"{timeout_s}s",
    ]
    for attempt in range(retries + 1):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            LOG.info(f"☁️  Uploaded to {remote_path}")
            return True
        err = (result.stderr or "").strip()
        LOG.warning(f"☁️  GDrive upload failed (attempt {attempt + 1}/{retries + 1}): {err}")
        time.sleep(0.5)
    return False


class LocalCamera:
    def __init__(self):
        self.cap = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(0)
        return bool(self.cap and self.cap.isOpened())

    def read(self):
        if not self.cap:
            return False, None
        ok, frame = self.cap.read()
        return (ok and frame is not None), (frame if ok else None)

    def reopen(self):
        self.release()
        time.sleep(0.2)
        self.open()

    def release(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.cap = None


def main() -> int:
    p = argparse.ArgumentParser(description="Capture N photos at a fixed interval (no movement).")
    p.add_argument("--count", type=int, default=10, help="Number of photos to take.")
    p.add_argument("--interval", type=float, default=1.0, help="Seconds between photos.")
    p.add_argument("--dest", type=str, default=None, help="scp destination, e.g. user@mac:/Users/you/BrunoShots/")
    p.add_argument("--gdrive-dest", type=str, default=None, help="rclone dest, e.g. gdrive:BrunoShots/")
    p.add_argument("--use-mac-cam", action="store_true", help="Use the Mac built-in camera (for local testing).")
    p.add_argument("--ssh-port", type=int, default=22, help="SSH port for scp.")
    p.add_argument("--upload-retries", type=int, default=1, help="scp retry count.")
    p.add_argument("--upload-timeout", type=int, default=4, help="scp connect timeout in seconds.")
    args = p.parse_args()

    if args.count <= 0:
        LOG.error("--count must be > 0")
        return 2

    camera = LocalCamera() if args.use_mac_cam else ExternalCamera()
    if not camera.open():
        LOG.error("❌ Cannot start without camera")
        return 1
    if args.use_mac_cam:
        LOG.info("🎥 Using Mac built-in camera")

    last_good_frame = None
    LOG.info(f"📷 Capture-only started: count={args.count} interval={args.interval}s")
    if args.dest:
        LOG.info(f"📤 Upload enabled: {args.dest}")
    if args.gdrive_dest:
        LOG.info(f"☁️  GDrive upload enabled: {args.gdrive_dest}")

    try:
        for i in range(args.count):
            frame = read_or_reconnect(camera, last_good_frame)
            if frame is None:
                LOG.warning("⚠️  No frame available (retrying)")
                time.sleep(0.05)
                continue
            last_good_frame = frame
            img_path = paths.save_image_path("capture_only")
            if cv2.imwrite(str(img_path), frame):
                LOG.info(f"💾 Saved {i + 1}/{args.count}: {img_path}")
                if args.gdrive_dest:
                    upload_gdrive(str(img_path), args.gdrive_dest, args.upload_timeout, args.upload_retries)
                else:
                    upload_file(str(img_path), args.dest, args.ssh_port, args.upload_timeout, args.upload_retries)
            else:
                LOG.warning(f"⚠️  Save failed for {img_path}")
            if i < args.count - 1:
                time.sleep(max(args.interval, 0.0))
    finally:
        try:
            camera.release()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
