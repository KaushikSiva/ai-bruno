#!/usr/bin/env python3
"""
Standalone MasterPi battery monitor.

Reads battery voltage from the Hiwonder Board SDK and estimates percentage.
"""

import argparse
import datetime as dt
import sys
import time
from typing import Any, Optional

from bruno_core.config.env import get_env_float, load_env


def _resolve_board() -> Any:
    masterpi_path = "/home/pi/MasterPi"
    if masterpi_path not in sys.path:
        sys.path.append(masterpi_path)
    from common.ros_robot_controller_sdk import Board  # type: ignore

    return Board()


def _pick_battery_method(board: Any) -> Optional[str]:
    for name in ("getBattery", "get_battery", "getBatteryVoltage", "get_battery_voltage"):
        if hasattr(board, name):
            return name
    return None


def _to_voltage(raw: Any) -> Optional[float]:
    try:
        if raw is None:
            return None
        if isinstance(raw, (list, tuple)) and raw:
            raw = raw[0]
        if isinstance(raw, dict):
            raw = raw.get("voltage") or raw.get("battery") or raw.get("mv")
        v = float(raw)
        # Most MasterPi SDK variants return millivolts.
        if v > 100.0:
            return v / 1000.0
        return v
    except Exception:
        return None


def _to_percent(voltage_v: float, v_min: float, v_max: float) -> int:
    lo = min(v_min, v_max)
    hi = max(v_min, v_max)
    if hi <= lo:
        return 0
    ratio = (voltage_v - lo) / (hi - lo)
    return int(round(max(0.0, min(1.0, ratio)) * 100.0))


def _read_once(board: Any, method_name: str, v_min: float, v_max: float) -> Optional[str]:
    raw = getattr(board, method_name)()
    voltage_v = _to_voltage(raw)
    if voltage_v is None:
        return None
    pct = _to_percent(voltage_v, v_min, v_max)
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{ts}  battery={voltage_v:.2f}V  est={pct}%"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MasterPi battery status")
    p.add_argument("--min-v", type=float, default=get_env_float("BRUNO_BATTERY_MIN_V", 10.2))
    p.add_argument("--max-v", type=float, default=get_env_float("BRUNO_BATTERY_MAX_V", 12.6))
    p.add_argument("--watch", action="store_true", help="Continuously print battery status")
    p.add_argument("--interval", type=float, default=2.0, help="Watch interval seconds")
    return p.parse_args()


def main() -> int:
    load_env()
    args = parse_args()
    try:
        board = _resolve_board()
    except Exception as exc:
        print(f"Failed to initialize Board SDK: {exc}")
        return 1

    method_name = _pick_battery_method(board)
    if not method_name:
        print("No battery API found on Board() (tried getBattery/get_battery/getBatteryVoltage/get_battery_voltage)")
        return 2

    if not args.watch:
        line = _read_once(board, method_name, args.min_v, args.max_v)
        if not line:
            print("Battery read failed")
            return 3
        print(line)
        return 0

    print(f"Using Board.{method_name}() min_v={args.min_v:.2f} max_v={args.max_v:.2f}")
    try:
        while True:
            line = _read_once(board, method_name, args.min_v, args.max_v)
            if line:
                print(line)
            else:
                print("Battery read failed")
            time.sleep(max(0.2, args.interval))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
