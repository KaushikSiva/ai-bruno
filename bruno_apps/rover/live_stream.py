#!/usr/bin/env python3
import os, time, cv2, numpy as np
from flask import Flask, Response
from bruno_core.config.env import get_env_int, get_env_str
app = Flask(__name__)

def src():
    s = get_env_str("STREAM_SOURCE")
    try: return int(s)
    except: return s

def open_cap():
    c = cv2.VideoCapture(src(), cv2.CAP_V4L2)
    if not c.isOpened(): c = cv2.VideoCapture(src())
    return c if c.isOpened() else None

def gen():
    cap = open_cap()
    while True:
        if cap is None or not cap.isOpened():
            frame = np.zeros((480,640,3), np.uint8); time.sleep(0.1)
        else:
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release(); cap = open_cap(); continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok: continue
        b = buf.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
               str(len(b)).encode() + b"\r\n\r\n" + b + b"\r\n")

@app.route("/")
def index():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def run_stream(host: str = "0.0.0.0", port: int = 8080):
    app.run(host=host, port=port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    host = get_env_str("HOST")
    port = get_env_int("PORT")
    run_stream(host=host, port=port)
