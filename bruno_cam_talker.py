import argparse
import base64
import os
import queue
import subprocess
import textwrap
import threading
import time
from typing import List

import cv2
import numpy as np
import requests
from pydub import AudioSegment


SUMMARY_FRAME_INTERVAL = 150
HYPERBOLIC_MODEL = os.getenv("HYPERBOLIC_MODEL", "phi-3-vision")
HYPERBOLIC_ENDPOINT = "https://api.hyperbolic.xyz/v1/chat/completions"
CAPTION_PROMPT = "Describe this image briefly in one sentence."
OBJECT_PROMPT = (
    "List all visible objects. Return only a comma-separated list of lowercase object names, no extra words."
)
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
TTS_OUTPUT_PATH = "tts_output.mp3"
AMPLITUDE_CHUNK_MS = 80
WINDOW_TITLE = "Bruno Vision UI"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bruno camera streamer with Hyperbolic vision and ElevenLabs TTS.")
    parser.add_argument("--stream-url", type=str, help="HTTP stream URL for Bruno's camera.")
    parser.add_argument("--camera-index", type=int, default=0, help="Local USB webcam index (default: 0).")
    return parser.parse_args()


def encode_frame_to_base64(frame: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise ValueError("Failed to encode frame to JPEG.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def hyperbolic_request(frame: np.ndarray, prompt: str) -> str:
    api_key = os.getenv("HYPERBOLIC_API_KEY")
    if not api_key:
        print("HYPERBOLIC_API_KEY not set; skipping vision call.")
        return ""

    try:
        image_b64 = encode_frame_to_base64(frame)
    except ValueError as exc:
        print(f"Vision encoding error: {exc}")
        return ""

    payload = {
        "model": HYPERBOLIC_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "media_type": "image/jpeg", "data": image_b64},
                ],
            }
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = requests.post(HYPERBOLIC_ENDPOINT, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        print(f"Hyperbolic request failed: {exc}")
        return ""

    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message", {})
    content = message.get("content", "")

    if isinstance(content, list):
        texts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "output_text"]
        return " ".join(t.strip() for t in texts if t.strip())
    if isinstance(content, str):
        return content.strip()
    return ""


def hyperbolic_caption(frame: np.ndarray) -> str:
    return hyperbolic_request(frame, CAPTION_PROMPT)


def hyperbolic_detect_objects(frame: np.ndarray) -> List[str]:
    result = hyperbolic_request(frame, OBJECT_PROMPT)
    if not result:
        return []
    objects = []
    for item in result.split(","):
        name = item.strip().lower()
        if name:
            objects.append(name)
    return objects


def build_summary(caption: str, objects: List[str], bottle_detected: bool) -> str:
    caption_text = caption.strip() if caption else "Unable to describe the scene."
    objects_text = ", ".join(objects) if objects else "unknown"
    summary = f"Objects: {objects_text}. Scene: {caption_text}"
    if bottle_detected:
        summary = summary.rstrip(".")
        summary += ". I also see a bottle."
    return summary


def update_amplitude(amplitude_state: dict, value: float) -> None:
    with amplitude_state["lock"]:
        amplitude_state["value"] = max(0.0, min(1.0, value))


def read_amplitude(amplitude_state: dict) -> float:
    with amplitude_state["lock"]:
        return amplitude_state["value"]


def perform_tts_request(text: str) -> bool:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("ELEVENLABS_API_KEY not set; cannot run TTS.")
        return False

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.3, "similarity_boost": 0.8},
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
    except Exception as exc:
        print(f"ElevenLabs request failed: {exc}")
        return False

    try:
        with open(TTS_OUTPUT_PATH, "wb") as f:
            f.write(response.content)
    except OSError as exc:
        print(f"Failed to write TTS output: {exc}")
        return False

    return True


def play_audio_with_amplitude(amplitude_state: dict) -> None:
    try:
        audio = AudioSegment.from_file(TTS_OUTPUT_PATH)
    except Exception as exc:
        print(f"Unable to process audio for visualization: {exc}")
        audio = None

    process = None
    try:
        process = subprocess.Popen(["afplay", TTS_OUTPUT_PATH])
    except Exception as exc:
        print(f"Audio playback failed: {exc}")
        update_amplitude(amplitude_state, 0.0)
        return

    if audio is not None:
        max_amp = audio.max_possible_amplitude or 1
        for start_ms in range(0, len(audio), AMPLITUDE_CHUNK_MS):
            if process.poll() is not None:
                break
            chunk = audio[start_ms : start_ms + AMPLITUDE_CHUNK_MS]
            amplitude = chunk.rms / max_amp if max_amp else 0.0
            update_amplitude(amplitude_state, amplitude)
            time.sleep(AMPLITUDE_CHUNK_MS / 1000.0)

    process.wait()
    update_amplitude(amplitude_state, 0.0)


def tts_thread_worker(text_queue: queue.Queue, amplitude_state: dict, stop_event: threading.Event) -> None:
    while not stop_event.is_set() or not text_queue.empty():
        try:
            text = text_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if text is None:
            text_queue.task_done()
            break

        if perform_tts_request(text):
            play_audio_with_amplitude(amplitude_state)
        text_queue.task_done()

    update_amplitude(amplitude_state, 0.0)


def enqueue_tts(text_queue: queue.Queue, text: str) -> None:
    text_queue.put(text)


def draw_overlays(frame: np.ndarray, summary: str, objects: List[str], amplitude: float) -> np.ndarray:
    overlay_frame = frame.copy()
    h, w = overlay_frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    objects_text = f"objects: {', '.join(objects) if objects else 'unknown'}"
    cv2.putText(overlay_frame, objects_text, (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    summary_display = summary if summary else "Waiting for summary..."
    wrapped_summary = textwrap.wrap(summary_display, width=50) or [summary_display]
    y_offset = h - 60
    for line in wrapped_summary[:3]:
        cv2.putText(overlay_frame, line, (10, y_offset), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y_offset += 25

    bar_width = 200
    bar_height = 20
    start_x = w - bar_width - 20
    start_y = h - 40
    cv2.rectangle(overlay_frame, (start_x, start_y), (start_x + bar_width, start_y + bar_height), (255, 255, 255), 2)
    fill_width = int(bar_width * max(0.0, min(1.0, amplitude)))
    cv2.rectangle(
        overlay_frame,
        (start_x, start_y),
        (start_x + fill_width, start_y + bar_height),
        (0, 165, 255),
        thickness=-1,
    )
    cv2.putText(overlay_frame, "audio", (start_x, start_y - 10), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay_frame


def main() -> None:
    args = parse_args()
    stream_url = args.stream_url
    camera_index = args.camera_index

    use_stream = bool(stream_url)
    if use_stream:
        print(f"Using stream URL: {stream_url}")
        capture_source = stream_url
    else:
        print(f"Using camera index: {camera_index}")
        capture_source = camera_index

    print("Press 'q' to quit.")
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        print("Failed to open video source.")
        return

    text_queue = queue.Queue()
    amplitude_state = {"value": 0.0, "lock": threading.Lock()}
    stop_event = threading.Event()
    tts_thread = threading.Thread(
        target=tts_thread_worker,
        args=(text_queue, amplitude_state, stop_event),
        daemon=True,
    )
    tts_thread.start()

    frame_counter = 0
    latest_summary = ""
    last_spoken_summary = ""
    latest_objects: List[str] = []

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from source.")
                break

            frame_counter += 1

            if frame_counter % SUMMARY_FRAME_INTERVAL == 0:
                caption = hyperbolic_caption(frame)
                objects = hyperbolic_detect_objects(frame)
                latest_objects = objects
                bottle_detected = "bottle" in latest_objects

                summary = build_summary(caption, latest_objects, bottle_detected)
                latest_summary = summary

                if bottle_detected:
                    print("[BOTTLE DETECTED] I see a bottle in the frame.")

                if summary and summary != last_spoken_summary:
                    enqueue_tts(text_queue, summary)
                    last_spoken_summary = summary

            amplitude = read_amplitude(amplitude_state)
            frame_with_overlay = draw_overlays(frame, latest_summary, latest_objects, amplitude)
            cv2.imshow(WINDOW_TITLE, frame_with_overlay)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        stop_event.set()
        text_queue.put(None)
        tts_thread.join(timeout=5)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
