#!/usr/bin/env python3
"""
Diagnose and fix color detection when the pick-place search finds nothing.

`search` failing tells you only that no blob passed both the HSV range and the
size filter — not which one rejected it. This reports each separately, then
samples the bottle itself to build a range that works.

Run on the robot with the bottle centered in the camera view:

    # what does each configured color actually match right now?
    python3 bruno_apps/pick_place/tune_detection.py

    # sample the bottle under the crosshair and print a suggested range
    python3 bruno_apps/pick_place/tune_detection.py --sample green

    # ...and write that range into config/bruno_config.json
    python3 bruno_apps/pick_place/tune_detection.py --sample green --save

Sampling reads a small patch at the center of the frame, so put the bottle there
and keep the lighting the same as when you run the app — HSV ranges tuned under
different light will not transfer.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bruno_core.camera.factory import make_camera, read_or_reconnect  # noqa: E402
from bruno_core.config.env import get_env_str, load_env  # noqa: E402
from bruno_core.logging.setup import LOG  # noqa: E402

CONFIG_PATH = os.path.join(REPO_ROOT, "config", "bruno_config.json")
PATCH = 40  # side of the center sampling square, pixels


def load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r") as fh:
            return json.load(fh)
    except Exception as exc:
        LOG.warning("Could not read %s: %s", CONFIG_PATH, exc)
        return {}


def config_ranges(cfg: dict) -> dict:
    detection = (cfg or {}).get("detection", {}) or {}
    out = {}
    for name, bounds in (detection.get("color_detection", {}) or {}).items():
        try:
            out[name] = (
                np.array(bounds["lower_hsv"], dtype=np.uint8),
                np.array(bounds["upper_hsv"], dtype=np.uint8),
            )
        except Exception:
            continue
    return out


def size_filter(cfg: dict) -> tuple:
    sf = ((cfg or {}).get("detection", {}) or {}).get("size_filter", {}) or {}
    return (
        int(sf.get("min_area", 500)),
        int(sf.get("max_area", 150000)),
        float(sf.get("min_aspect_ratio", 1.2)),
        float(sf.get("max_aspect_ratio", 4.5)),
    )


def report(frame, ranges: dict, min_area: int, max_area: int, min_ar: float, max_ar: float) -> None:
    """Show what each configured range matches, and why blobs get rejected."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)
    total = frame.shape[0] * frame.shape[1]
    print(
        f"\nframe {frame.shape[1]}x{frame.shape[0]}, size filter {min_area}-{max_area} px², "
        f"aspect (height/width) {min_ar}-{max_ar}"
    )
    for name, (lower, upper) in ranges.items():
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        matched = int(np.count_nonzero(mask))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
        pct = 100.0 * matched / total
        print(f"\n  {name}: {matched} px matched ({pct:.1f}% of frame)")
        if not blobs:
            print("    no contours — the HSV range matches nothing in this frame")
            continue
        for contour in blobs:
            area = cv2.contourArea(contour)
            _, _, w, h = cv2.boundingRect(contour)
            aspect = h / float(w) if w > 0 else 0.0
            if area < min_area:
                verdict = f"REJECTED, below min_area {min_area}"
            elif area > max_area:
                verdict = f"REJECTED, above max_area {max_area}"
            elif not (min_ar <= aspect <= max_ar):
                verdict = f"REJECTED, aspect {aspect:.2f} outside {min_ar}-{max_ar} (not bottle-shaped)"
            else:
                verdict = "would be ACCEPTED"
            print(f"    blob {area:8.0f} px² {w}x{h} aspect {aspect:.2f} — {verdict}")


def sample(frame, color: str) -> tuple:
    """Read the center patch and derive an HSV range covering it."""
    h, w = frame.shape[:2]
    cy, cx = h // 2, w // 2
    half = PATCH // 2
    patch = frame[cy - half : cy + half, cx - half : cx + half]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    # Percentiles rather than min/max: the patch edges catch background pixels,
    # and one stray corner should not widen the range across the spectrum.
    lo = np.percentile(pixels, 10, axis=0)
    hi = np.percentile(pixels, 90, axis=0)
    median = np.median(pixels, axis=0)

    # Pad hue tightly and saturation/value loosely — lighting moves S and V far
    # more than it moves hue.
    lower = np.array(
        [max(0, lo[0] - 10), max(40, lo[1] - 60), max(40, lo[2] - 60)], dtype=np.uint8
    )
    upper = np.array(
        [min(180, hi[0] + 10), min(255, hi[1] + 60), min(255, hi[2] + 60)], dtype=np.uint8
    )
    print(f"\ncenter patch {PATCH}x{PATCH} px, median HSV = {median.astype(int).tolist()}")
    print(f"suggested {color}: lower_hsv={lower.tolist()} upper_hsv={upper.tolist()}")
    return lower, upper


def save_range(color: str, lower, upper) -> None:
    cfg = load_config()
    colors = cfg.setdefault("detection", {}).setdefault("color_detection", {})
    # Detection strips at the underscore ("green_plastic" -> "green"), so keep
    # whichever existing key maps to this color rather than adding a duplicate.
    key = next((k for k in colors if k.split("_")[0].lower() == color), f"{color}_plastic")
    colors[key] = {"lower_hsv": [int(v) for v in lower], "upper_hsv": [int(v) for v in upper]}
    with open(CONFIG_PATH, "w") as fh:
        json.dump(cfg, fh, indent=4)
    print(f"\n💾 saved {key} -> {CONFIG_PATH}")


def main() -> int:
    load_env(os.path.join(REPO_ROOT, ".env"))
    parser = argparse.ArgumentParser(description="Diagnose and tune pick-place color detection")
    parser.add_argument("--mode", default=get_env_str("CAM_MODE", "builtin"), choices=["builtin", "external"])
    parser.add_argument("--sample", metavar="COLOR", default=None, help="derive a range from the center patch")
    parser.add_argument("--save", action="store_true", help="write the sampled range to config")
    parser.add_argument("--frames", type=int, default=5, help="frames to skip before measuring (let AE settle)")
    args = parser.parse_args()

    if args.save and not args.sample:
        parser.error("--save needs --sample COLOR")

    camera = make_camera(args.mode, retry_attempts=3, retry_delay=2.0)
    frame = None
    for _ in range(max(1, args.frames)):
        frame = read_or_reconnect(camera)
    if frame is None:
        print("no frame from camera")
        return 1

    cfg = load_config()
    min_area, max_area, min_ar, max_ar = size_filter(cfg)
    report(frame, config_ranges(cfg), min_area, max_area, min_ar, max_ar)

    if args.sample:
        lower, upper = sample(frame, args.sample)
        if args.save:
            save_range(args.sample, lower, upper)
        else:
            print("(re-run with --save to write this into config/bruno_config.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
