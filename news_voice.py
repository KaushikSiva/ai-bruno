#!/usr/bin/env python3
"""
Lightweight voice-triggered news player for a Raspberry Pi touchscreen.

Listens for phrases like "what's the latest news" and opens the configured
YouTube video full screen in Chromium. An HTTP status endpoint reports the
listening state so you can verify the service remotely.
"""

from __future__ import annotations

import argparse
import logging
import queue
import subprocess
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import speech_recognition as sr

NEWS_VIDEO_URL = "https://youtu.be/jTBJT7WYusM"
NEWS_PHRASES = (
    "latest news",
    "what's the latest news",
    "whats the latest news",
    "what is the latest news",
    "tell me the news",
)
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
MIN_XAI_WORDS = 5
SILENCE_TIMEOUT_SEC = 30.0
CHROMIUM_ARGS = (
    "chromium-browser",
    "--noerrdialogs",
    "--disable-infobars",
    "--disable-translate",
    "--no-first-run",
    "--no-default-browser-check",
    "--force-device-scale-factor=1",
    "--overscroll-history-navigation=0",
    "--window-size=800,480",
    "--disable-gpu",
    "--disable-gpu-compositing",
    "--use-gl=egl",
    "--disk-cache-dir=/tmp/chrome-cache",
    "--disk-cache-size=1",
    "--enable-unsafe-swiftshader",
    "--disable-software-rasterizer",
    "--disable-accelerated-video",
    "--disable-gpu-vsync",
    "--autoplay-policy=no-user-gesture-required",
)
RECOGNITION_TIMEOUT = 1.0


class NewsHTTPServer(HTTPServer):
    allow_reuse_address = True


class StatusHandler(BaseHTTPRequestHandler):
    server_version = "NewsVoiceServer/0.1"
    sys_version = ""

    def do_GET(self) -> None:
        if self.path not in ("/", "/status"):
            self.send_error(404, "Use / or /status")
            return

        with state_lock:
            payload = (
                "Listening for phrases:\n"
                f"  {', '.join(NEWS_PHRASES)}\n\n"
                f"Last phrase heard: {state['last_phrase'] or 'none yet'}\n"
                f"Last heard at: {state['last_heard'] or 'n/a'}\n"
                f"Last played at: {state['last_played'] or 'n/a'}\n"
            )

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(payload.encode("utf-8"))

    def log_message(self, format: str, *args) -> None:  # type: ignore[override]
        # Silence the default request logging to keep stdout clean.
        logging.debug("HTTP %s - %s", self.client_address[0], format % args)


state_lock = threading.Lock()
state = {
    "last_phrase": None,
    "last_heard": None,
    "last_played": None,
}

player_lock = threading.Lock()
player_process: subprocess.Popen | None = None
NEWS_COMMAND = "play_news"
CALL_DUMMY_COMMAND = "call_dummy"

XAI_RESPONSE_VIBE = "Bring humor and warmth while staying helpful."
XAI_RESPONSE_FILENAME = "news_voice_xai_response.mp3"
TTS_PLAYER_ARGS = ("mpg123", "-q")
CommandEntry = tuple[str, str]

try:
    from xai_assistant import respond_to_prompt
except ImportError:  # pragma: no cover - optional dependency
    respond_to_prompt = None

def run_status_server(host: str, port: int, stop_event: threading.Event) -> None:
    with NewsHTTPServer((host, port), StatusHandler) as server:
        logging.info("Status server is up at http://%s:%d/status", host, port)
        server.timeout = 1.0
    last_heard_time = time.time()
    while not stop_event.is_set():
        if time.time() - last_heard_time > SILENCE_TIMEOUT_SEC:
            logging.info("No speech detected for %s seconds; shutting down", SILENCE_TIMEOUT_SEC)
            stop_event.set()
            return
            server.handle_request()


def does_phrase_match(text: str) -> bool:
    normalized = text.lower()
    return any(phrase in normalized for phrase in NEWS_PHRASES)


def recognize_audio(recognizer: sr.Recognizer, audio: sr.AudioData) -> str | None:
    try:
        return recognizer.recognize_sphinx(audio)
    except (sr.UnknownValueError, sr.RequestError) as exc:
        logging.debug("Sphinx not ready or could not understand: %s", exc)
    except AttributeError as exc:
        logging.debug("PocketSphinx missing: %s", exc)

    try:
        return recognizer.recognize_google(audio)
    except (sr.UnknownValueError, sr.RequestError) as exc:
        logging.debug("Google recognizer failed (network may be disabled): %s", exc)

    return None


def listen_for_news(
    stop_event: threading.Event,
    command_queue: queue.Queue[CommandEntry],
) -> None:
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        logging.info("Calibrating mic for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1.0)

        while not stop_event.is_set():
            try:
                audio = recognizer.listen(
                    source,
                    timeout=RECOGNITION_TIMEOUT,
                    phrase_time_limit=6,
                )
            except sr.WaitTimeoutError:
                continue

            transcription = recognize_audio(recognizer, audio)
            if not transcription:
                continue

            logging.info("Heard: %s", transcription)

            last_heard_time = time.time()

            with state_lock:
                state["last_phrase"] = transcription
                state["last_heard"] = time.strftime("%Y-%m-%d %H:%M:%S")

            if does_phrase_match(transcription):
                logging.info("Trigger phrase detected; queueing news playback")
                command_queue.put_nowait((NEWS_COMMAND, transcription))
            else:
                words = [word for word in transcription.split() if word]
                if len(words) >= MIN_XAI_WORDS:
                    logging.info("Non-news phrase detected; queueing XAI response")
                    command_queue.put_nowait((CALL_DUMMY_COMMAND, transcription))
                else:
                    logging.info(
                        "Skipping XAI response for short phrase (%d words)",
                        len(words),
                    )


def _youtube_embed_url(url: str) -> str:
    return url


def build_player_command(url: str) -> tuple[list[str], str]:
    return [*CHROMIUM_ARGS, _youtube_embed_url(url)], "chromium"


def play_news(video_url: str) -> None:
    global player_process

    with player_lock:
        if player_process and player_process.poll() is None:
            logging.info("Player already running; skipping new launch")
            return

        try:
            command, label = build_player_command(video_url)
            logging.info("Launching %s full-screen for %s", label, video_url)
            player_process = subprocess.Popen(
                command,
                start_new_session=True,
            )
        except FileNotFoundError:
            logging.error("Chromium not found; install it before running this script")
            return

    with state_lock:
        state["last_played"] = time.strftime("%Y-%m-%d %H:%M:%S")


def _xai_output_path() -> Path:
    return Path(tempfile.gettempdir()) / XAI_RESPONSE_FILENAME


def _play_xai_audio(audio_path: Path) -> None:
    command = [*TTS_PLAYER_ARGS, str(audio_path)]
    try:
        logging.info("Playing XAI response via %s", " ".join(command))
        subprocess.run(command, check=False)
    except FileNotFoundError:
        logging.error("mpg123 not found; install it to hear XAI assistant responses")
    except subprocess.SubprocessError as exc:
        logging.warning("Failed to play XAI audio: %s", exc)


def _respond_with_xai(prompt_text: str, vibe: str) -> None:
    if not respond_to_prompt:
        logging.debug("XAI assistant helper not available; skipping response")
        return

    try:
        audio_path = respond_to_prompt(prompt_text, vibe, _xai_output_path())
    except Exception as exc:
        logging.warning("XAI assistant request failed: %s", exc)
        return

    _play_xai_audio(audio_path)


def handle_non_news_phrase(prompt_text: str, vibe: str) -> None:
    def _worker() -> None:
        _respond_with_xai(prompt_text, vibe)

    thread = threading.Thread(
        target=_worker,
        name="xai-response",
        daemon=True,
    )
    thread.start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice-triggered news launcher")
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Interface for the status HTTP server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port for the status server (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--video-url",
        default=NEWS_VIDEO_URL,
        help="YouTube URL that should play when triggered",
    )
    parser.add_argument(
        "--xai-vibe",
        default=XAI_RESPONSE_VIBE,
        help="Instructions passed to the XAI assistant when crafting a spoken reply",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    stop_event = threading.Event()
    command_queue: queue.Queue[CommandEntry] = queue.Queue()

    listener = threading.Thread(
        target=listen_for_news,
        args=(stop_event, command_queue),
        name="listener",
        daemon=True,
    )
    listener.start()

    server_thread = threading.Thread(
        target=run_status_server,
        args=(args.host, args.port, stop_event),
        name="status-server",
        daemon=True,
    )
    server_thread.start()

    logging.info("News voice server running")

    try:
        while not stop_event.is_set():
            try:
                command, payload = command_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if command == NEWS_COMMAND:
                play_news(args.video_url)
            elif command == CALL_DUMMY_COMMAND:
                handle_non_news_phrase(payload, args.xai_vibe)
            else:
                logging.warning("Unknown command from listener: %s", command)

    except KeyboardInterrupt:
        logging.info("Shutting down on keyboard interrupt")
        stop_event.set()

    listener.join(timeout=2.0)
    server_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
