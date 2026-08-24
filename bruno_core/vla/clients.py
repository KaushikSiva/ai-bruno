"""Dependency-free HTTP client for the robot bridge."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Mapping, Optional

from .contracts import Action


class HttpServiceError(RuntimeError):
    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(f"bridge returned HTTP {status}: {detail}")


def request_json(
    method: str,
    url: str,
    *,
    payload: Optional[Mapping[str, Any]] = None,
    headers: Optional[Mapping[str, str]] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request_headers = {"Accept": "application/json", **dict(headers or {})}
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=request_headers, method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw, status = response.read(), response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            parsed = json.loads(raw)
            detail = str(parsed.get("error", parsed)) if isinstance(parsed, dict) else str(parsed)
        except Exception:
            detail = raw.decode("utf-8", errors="replace")
        raise HttpServiceError(exc.code, detail) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach {url}: {exc.reason}") from exc
    if not 200 <= status < 300:
        raise HttpServiceError(status, raw.decode("utf-8", errors="replace"))
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{url} did not return valid JSON") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{url} did not return a JSON object")
    return data


class RobotClient:
    """Talks to a `bruno_core.vla.server` bridge over HTTP."""

    def __init__(self, base_url: str = "http://127.0.0.1:8091", token: str = "", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}

    def status(self) -> Dict[str, Any]:
        return request_json("GET", f"{self.base_url}/v1/status", headers=self._headers, timeout=self.timeout)

    def telemetry(self) -> Dict[str, Any]:
        return request_json("GET", f"{self.base_url}/v1/telemetry", headers=self._headers, timeout=self.timeout)

    def calibration(self) -> Dict[str, Any]:
        return request_json("GET", f"{self.base_url}/v1/calibration", headers=self._headers, timeout=self.timeout)

    def set_armed(self, armed: bool) -> Dict[str, Any]:
        return request_json(
            "POST", f"{self.base_url}/v1/arm", payload={"armed": bool(armed)},
            headers=self._headers, timeout=self.timeout,
        )

    def command(self, action: Action) -> Dict[str, Any]:
        return request_json(
            "POST", f"{self.base_url}/v1/command", payload=action.to_dict(),
            headers=self._headers, timeout=self.timeout,
        )
