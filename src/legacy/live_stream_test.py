#!/usr/bin/env python3
# coding: utf-8
"""
Safe MJPEG streamer for USB/RTSP cameras
- Uses STREAM_SOURCE env (0, 1, /dev/videoX, or rtsp://...)
- Never raises RuntimeError if camera missing: falls back to black frames
- Accessible at http://<pi-ip>:8080/
"""

import os, time
import cv2
import numpy as np
from flask import Flask, Response

app = Flask(__name__)

def _parse_source(val: str):
    try:
        return int(val)
    except Exception:
        return val

SOURCE = _parse_source(os.environ.get("STREAM_SOURCE", "0"))

CAP_PROPS = {
    cv2.CAP_PROP_FRAME_WIDTH: int(os.environ.get("STREAM_WIDTH", "1280")),
    cv2.CAP_PROP_FRAME_HEIGHT: int(os.environ.get("STREAM_HEIGHT", "720")),
    cv2.CAP_PROP_FPS: int(os.environ.get("STREAM_FPS", "30")),
}

def open_capture(source):
    cap = None
    if isinstance(source, int):
        # Try V4L2 first, fallback to default
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

    if not cap or not cap.isOpened():
        return None

    for k, v in CAP_PROPS.items():
        cap.set(k, v)
    return cap

def mjpeg_generator():
    cap = open_capture(SOURCE)
    while True:
        if cap is None or not cap.isOpened():
            # Fallback: serve black frame if no camera
            frame = np.zeros((480, 640, 3), np.uint8)
            time.sleep(0.1)
        else:
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                cap = open_capture(SOURCE)
                continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        jpg = buf.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
               jpg + b"\r\n")

@app.route("/")
def index():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def run_stream(host="0.0.0.0", port=8080):
    app.run(host=host, port=port, threaded=True, use_reloader=False)

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    run_stream(host, port)
