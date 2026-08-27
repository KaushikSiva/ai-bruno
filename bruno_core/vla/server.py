"""Token-authenticated HTTP bridge in front of one motion driver.

Run this next to the hardware it drives -- on Bruno for `--driver bruno`, on
the workstation for `--driver mujoco` -- and point teleop at it. The bridge is
the only process that touches the robot, it starts disarmed, and its dead-man
timer stops the robot if the client goes away.

    python3 -m bruno_core.vla.server --driver mujoco --mujoco-viewer
    BRUNO_SHARED_TOKEN=... python3 -m bruno_core.vla.server --driver bruno --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
from http.server import ThreadingHTTPServer
from typing import Type
from urllib.parse import urlsplit

from .calibration import load_profile
from .contracts import Action, ContractError
from .controller import RobotController, RobotNotArmed, limits_from_env
from .drivers import make_driver
from .http_utils import JsonHandler

LOG = logging.getLogger("bruno_core.vla.server")
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def build_controller(args: argparse.Namespace) -> RobotController:
    profile = load_profile(args.config)
    driver = make_driver(
        args.driver,
        profile=profile,
        model_path=args.mujoco_model,
        dry_run=args.dry_run,
        allow_uncalibrated=args.allow_uncalibrated,
    )
    return RobotController(driver=driver, profile=profile, **limits_from_env())


def make_handler(controller: RobotController, token: str) -> Type[JsonHandler]:
    class BridgeHandler(JsonHandler):
        auth_token = token

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/health":
                self.send_json(200, {"ok": True, "service": "bruno-vla-bridge"})
                return
            if not self.require_auth():
                return
            if path == "/v1/status":
                self.send_json(200, controller.status())
            elif path == "/v1/calibration":
                self.send_json(200, controller.profile.to_dict())
            elif path == "/v1/telemetry":
                telemetry = getattr(controller.driver, "telemetry", None)
                if telemetry is None:
                    self.send_json(404, {"error": "this driver has no telemetry"})
                else:
                    self.send_json(200, {"telemetry": telemetry()})
            elif path == "/v1/camera":
                self._send_camera()
            else:
                self.send_json(404, {"error": "not found"})

        def _send_camera(self) -> None:
            render = getattr(controller.driver, "render_camera_jpeg", None)
            if render is None:
                self.send_json(404, {"error": "this driver has no camera"})
                return
            try:
                body = render()
            except Exception as exc:
                LOG.exception("Camera render failed")
                self.send_json(503, {"error": f"camera failure: {type(exc).__name__}"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            if path not in {"/v1/arm", "/v1/command"}:
                self.send_json(404, {"error": "not found"})
                return
            if not self.require_auth():
                return
            try:
                payload = self.read_json()
                if path == "/v1/arm":
                    if not isinstance(payload.get("armed"), bool):
                        raise ContractError("armed must be true or false")
                    result = controller.set_armed(payload["armed"])
                else:
                    result = controller.execute(Action.from_dict(payload))
            except RobotNotArmed as exc:
                self.send_json(409, {"error": str(exc), "armed": False})
            except (ValueError, ContractError) as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception as exc:
                LOG.exception("Bridge command failed")
                self.send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
            else:
                self.send_json(200, result)

    return BridgeHandler


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Bruno VLA robot bridge")
    parser.add_argument("--driver", choices=("mock", "mujoco", "bruno"),
                        default=os.getenv("BRUNO_ROBOT_DRIVER", "mock"),
                        help="mock records commands; mujoco simulates; bruno moves the robot")
    parser.add_argument("--host", default=os.getenv("BRUNO_BRIDGE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=_env_int("BRUNO_BRIDGE_PORT", 8091))
    parser.add_argument("--config", default="", help="Path to bruno_config.json")
    parser.add_argument("--mujoco-model", default="", help="Override the MasterPi MJCF path")
    parser.add_argument("--mujoco-viewer", action="store_true",
                        help="Open the interactive viewer (macOS: run with mjpython)")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --driver bruno, log motions instead of executing them")
    parser.add_argument("--allow-uncalibrated", action="store_true",
                        help="Permit joint-space arm commands with an unverified pulse map")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    token = os.getenv("BRUNO_SHARED_TOKEN", "")
    if not token and args.host not in LOOPBACK:
        raise SystemExit("BRUNO_SHARED_TOKEN is required when the bridge is exposed on the network")
    if args.mujoco_viewer and args.driver != "mujoco":
        raise SystemExit("--mujoco-viewer requires --driver mujoco")

    controller = build_controller(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(controller, token))
    server.logger = LOG  # type: ignore[attr-defined]
    LOG.info(
        "Bridge listening on http://%s:%d driver=%s (starts disarmed)",
        args.host, args.port, args.driver,
    )
    if args.mujoco_viewer:
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.25},
            name="bruno-vla-http", daemon=True,
        )
        thread.start()
        try:
            controller.driver.run_viewer()  # type: ignore[attr-defined]
        except KeyboardInterrupt:
            LOG.info("Stopping viewer and bridge")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1.0)
            controller.close()
    else:
        try:
            server.serve_forever(poll_interval=0.25)
        except KeyboardInterrupt:
            LOG.info("Stopping bridge")
        finally:
            server.server_close()
            controller.close()


if __name__ == "__main__":
    main()
