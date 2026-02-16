# News Voice Server

`news_voice.py` is a minimal voice-triggered service that listens on the Raspberry Pi microphone for phrases such as _"what’s the latest news"_ and immediately launches the configured YouTube video (`https://youtu.be/jTBJT7WYusM` by default) in Chromium by navigating straight to the provided URL. The default Chromium arguments (`--window-size=800,480`, `--force-device-scale-factor=1`, `--overscroll-history-navigation=0`, `--disable-gpu`, `--disable-gpu-compositing`, `--use-gl=egl`, `--disk-cache-dir=/tmp/chrome-cache`, `--disk-cache-size=1`, `--enable-unsafe-swiftshader`, `--disable-software-rasterizer`, `--disable-accelerated-video`, `--disable-gpu-vsync`, `--autoplay-policy=no-user-gesture-required`) keep the window sized for the touchscreen even when hardware decode fails, and allow autoplay even though Chromium normally needs a user gesture. A simple HTTP endpoint (`/status`) reports the last phrase that was captured and the latest playback timestamp, which helps verification on a headless network or when the small 5" screen is showing the video.

## Dependencies

- Python packages (install with `pip install -r requirements.txt`)
  - `SpeechRecognition`
  - `PyAudio`
  - `pocketsphinx`
  - `opencv-python`, `numpy`, `pydub`, `requests` (already part of the repository requirements)
- `Chromium Browser` (install via `sudo apt install chromium-browser` so the default player is available)
- `mpg123` (install via `sudo apt install mpg123`) so the Grok assistant’s MP3 response can be heard.

## Running

1. Activate your Python environment (if you use one) and install requirements:

   ```bash
   pip install -r requirements.txt
   ```

2. Plug in and configure the microphone so the Pi treats it as the default capture device (check `arecord -l` if unsure).

3. Launch the listener:

   ```bash
   python news_voice.py --host 0.0.0.0 --port 8080
   ```

   The service will log what it hears and triggers Chromium when a matching phrase is detected. Point a browser to `http://<pi-ip>:8080/status` to see the listener state.

    - Chromium is already the default kiosk browser, so no extra flags are needed. Make sure `chromium-browser` is installed (`sudo apt install chromium-browser`).

4. Speak a phrase such as _"what’s the latest news"_ within range of the microphone. The server will queue Chromium, skip repeating the command while playback is still running, and keep the screen focused on the embed player. Any other recognized query with at least five words is sent to the Grok assistant so you hear a spoken response before the listener resumes; shorter snippets are ignored to avoid extra chatter. The process automatically exits if it hears nothing for 30 seconds so the Pi can reclaim the small touchscreen for other uses.

## Screen notes

- The 5" display will show the YouTube video in full screen via Chromium. Make sure nothing else grabs the HDMI output when the video is running.
- If you want to change the video, update the `--video-url` argument or modify the default constant in `news_voice.py`.
