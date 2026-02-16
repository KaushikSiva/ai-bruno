#!/usr/bin/env python3
"""Simple XAI Grok chat+TTS assistant wrapper."""

from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
from textwrap import dedent

import requests
from xai_sdk import Client
from xai_sdk.chat import system, user

BASE_URL = "https://us-east-4.api.x.ai/voice-staging"
ENDPOINT = f"{BASE_URL}/api/v1/text-to-speech/generate"
MAX_INPUT_LENGTH = 4096
MAX_PROMPT_LENGTH = 4096


def _grok_tts(output_path: Path, text: str, vibe: str) -> Path:
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY must be set to produce audio.")

    payload = {
        "model": "grok-voice",
        "input": text[:MAX_INPUT_LENGTH],
        "prompt": text[:MAX_PROMPT_LENGTH],
        "response_format": "mp3",
        "instructions": vibe,
        "sampling_params": {
            "max_new_tokens": 512,
            "temperature": 1.0,
            "min_p": 0.01,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.post(
        ENDPOINT,
        json=payload,
        stream=True,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Grok TTS failed ({response.status_code}): {response.text}"
        )

    with output_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                handle.write(chunk)

    print(f"Audio saved to {output_path}")
    return output_path


def respond_to_prompt(prompt_text: str, vibe: str, output_path: Path) -> Path:
    client = Client(api_key=os.getenv("XAI_API_KEY"), timeout=3600)
    chat = client.chat.create(model="grok-4")
    chat.append(system("You are a helpful and funny assistant."))
    chat.append(user(prompt_text))
    response = chat.sample()

    message = response.content.strip()
    print("Assistant:", message)
    return _grok_tts(output_path, message, vibe)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=dedent(
            """
            Query Grok-4 for a friendly/funny reply and render it as audio.
            """
        )
    )
    parser.add_argument(
        "--prompt",
        "-p",
        required=True,
        help="Text prompt for the assistant.",
    )
    parser.add_argument(
        "--vibe",
        "-v",
        default="Bring humor and warmth while staying helpful.",
        help="Instructions for the grok voice.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="outputs/assistant_response.mp3",
        help="Where to write the response audio.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    respond_to_prompt(args.prompt, args.vibe, Path(args.output))


if __name__ == "__main__":
    main()
