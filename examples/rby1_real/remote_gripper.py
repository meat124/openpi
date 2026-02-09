import json
import logging
import os
import socket
import threading
import time
from pathlib import Path

import numpy as np
import yaml


logger = logging.getLogger(__name__)


class Gripper:
    """UDP client for remote gripper control via JSON protocol.
    
    Protocol:
      - ping: health check
      - homing: calibrate gripper range
      - start/stop: control loop
      - set_normalized_target: send gripper command [right, left]
      - get_state: read current gripper position
    """

    GRIPPER_DIRECTION = False

    def __init__(self):
        """Load config from config.yaml and environment variables."""
        # Load from config.yaml
        host = None
        port = None
        timeout = None
        try:
            config_path = Path(__file__).resolve().parent / "config.yaml"
            with config_path.open(encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                host = cfg.get("remote_gripper_host", None)
                port = cfg.get("remote_gripper_port", None)
                timeout = cfg.get("remote_gripper_timeout", None)
        except Exception:
            logger.warning("Failed to load config.yaml")

        # Override with environment variables
        host = os.getenv("REMOTE_GRIPPER_HOST", host)
        port_env = os.getenv("REMOTE_GRIPPER_PORT", None)
        timeout_env = os.getenv("REMOTE_GRIPPER_TIMEOUT", None)
        if port_env is not None:
            try:
                port = int(port_env)
            except ValueError:
                logger.warning("Invalid REMOTE_GRIPPER_PORT: %s", port_env)
        if timeout_env is not None:
            try:
                timeout = float(timeout_env)
            except ValueError:
                logger.warning("Invalid REMOTE_GRIPPER_TIMEOUT: %s", timeout_env)

        self.host = host
        self.port = port
        self.timeout = timeout
        self.target_q: np.typing.NDArray = np.zeros(2, dtype=float)  # Cached normalized target

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock_lock = threading.Lock()
        if self.timeout is not None:
            try:
                self._sock.settimeout(float(self.timeout))
            except Exception:
                pass

    def _udp_request(self, payload: dict, expect_reply: bool) -> dict | None:
        """Send UDP request and optionally wait for reply."""
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        try:
            if self.host is None or self.port is None:
                logger.warning("Gripper host/port not configured")
                return None
            with self._sock_lock:
                self._sock.sendto(data, (self.host, self.port))
                if not expect_reply:
                    return None

                expected_cmd = payload.get("cmd", None)
                candidate = None

                # Wait up to socket timeout for matching response
                try:
                    sock_timeout = self._sock.gettimeout()
                except Exception:
                    sock_timeout = None
                if sock_timeout is None:
                    sock_timeout = 1.0
                deadline = time.monotonic() + float(sock_timeout)

                recv_count = 0
                max_reads = 100
                while time.monotonic() < deadline and recv_count < max_reads:
                    remaining = max(0.0, deadline - time.monotonic())
                    try:
                        self._sock.settimeout(remaining)
                        raw, _ = self._sock.recvfrom(65535)
                        recv_count += 1
                        try:
                            resp = json.loads(raw.decode("utf-8"))
                        except Exception:
                            continue

                        # If remote doesn't include cmd, fall back to first response.
                        resp_cmd = resp.get("cmd", None) if isinstance(resp, dict) else None
                        if expected_cmd is None or resp_cmd is None:
                            return resp

                        if resp_cmd == expected_cmd:
                            candidate = resp
                            # Drain any already-queued packets without blocking to get the latest.
                            prev_timeout = None
                            try:
                                prev_timeout = self._sock.gettimeout()
                            except Exception:
                                prev_timeout = None
                            try:
                                self._sock.settimeout(0.0)
                                for _ in range(max_reads - recv_count):
                                    try:
                                        raw2, _ = self._sock.recvfrom(65535)
                                        recv_count += 1
                                        try:
                                            resp2 = json.loads(raw2.decode("utf-8"))
                                        except Exception:
                                            continue
                                        if isinstance(resp2, dict) and resp2.get("cmd", None) == expected_cmd:
                                            candidate = resp2
                                    except (socket.timeout, BlockingIOError):
                                        break
                                    except Exception:
                                        break
                            finally:
                                try:
                                    self._sock.settimeout(prev_timeout)
                                except Exception:
                                    pass
                            return candidate
                        # else: ignore non-matching responses and keep reading
                    except socket.timeout:
                        break
                return candidate
        except Exception:
            return None

    def initialize(self, verbose=False):
        self._udp_request({"cmd": "initialize", "ts": time.time()}, expect_reply=False)
        resp = self._udp_request({"cmd": "ping", "ts": time.time()}, expect_reply=True)
        ok = bool(resp and resp.get("ok", False))
        if verbose:
            logger.info("[Gripper] Remote gripper ping (%s:%s) -> %s", self.host, self.port, ok)
        return ok

    def set_operating_mode(self, mode):
        resp = self._udp_request({"cmd": "set_operating_mode", "mode": mode, "ts": time.time()}, expect_reply=True)
        if not resp or not resp.get("ok", False):
            raise RuntimeError("[Gripper] Failed to set remote operating mode (no response or ok=false)")

    def homing(self):
        resp = self._udp_request({"cmd": "homing", "ts": time.time()}, expect_reply=True)
        ok = bool(resp and resp.get("ok", False))
        if not ok:
            logger.warning("[Gripper] Remote homing failed or no response")
            return False
        self.min_q = np.asarray(resp.get("min_q", None), dtype=float).reshape(-1)
        self.max_q = np.asarray(resp.get("max_q", None), dtype=float).reshape(-1)
        logger.info("[Gripper] Remote homing success. min_q: %s, max_q: %s", self.min_q, self.max_q)
        return True

    def start(self):
        resp = self._udp_request({"cmd": "start", "ts": time.time()},expect_reply=True)
        if not resp or not resp.get("ok", False):
            raise RuntimeError("[Gripper] Failed to start remote gripper loop (no response or ok=false)")

    def stop(self):
        resp = self._udp_request({"cmd": "stop", "ts": time.time()},expect_reply=True)
        if not resp or not resp.get("ok", False):
            raise RuntimeError("[Gripper] Failed to stop remote gripper loop (no response or ok=false)")

    def loop(self):
        pass

    def get_target(self):
        """Return cached target (fast, no network call)."""
        return self.target_q

    def get_target_remote(self):
        """Fetch target from remote server (slow, with network round-trip)."""
        resp = self._udp_request({"cmd": "get_target", "ts": time.time()}, expect_reply=True)
        if not resp or resp.get("target", None) is None:
            raise RuntimeError("Failed to get remote target")
        self.target_q = np.asarray(resp.get("target", None), dtype=float).reshape(-1)
        return self.target_q
    
    def get_normalized_target(self):
        """Fetch normalized target from remote server."""
        resp = self._udp_request({"cmd": "get_normalized_target", "ts": time.time()}, expect_reply=True)
        if not resp or not resp.get("ok", False):
            raise RuntimeError("Failed to fetch normalized target")
        target = resp.get("target", None)
        if target is None:
            raise RuntimeError("Normalized target missing in response")
        return np.asarray(target, dtype=float).reshape(-1)

    def set_normalized_target(self, normalized_q, wait_for_reply: bool = False):
        """Send normalized target to remote server.
        
        Args:
            normalized_q: Target values [right, left] in range [0, 1]
            wait_for_reply: If False (default), send fire-and-forget for low latency
        """
        normalized_q = np.asarray(normalized_q, dtype=float).reshape(-1)
        self.target_q = normalized_q  # Update cache immediately

        resp = self._udp_request(
            {"cmd": "set_normalized_target", "normalized_q": normalized_q.tolist(), "ts": time.time()},
            expect_reply=bool(wait_for_reply),
        )
        if wait_for_reply:
            if not resp or not resp.get("ok", False):
                raise RuntimeError("Failed to set normalized target")
            if resp.get("target", None) is not None:
                self.target_q = np.asarray(resp.get("target", None), dtype=float).reshape(-1)

    def get_state(self):
        """Read current gripper position from remote server."""
        resp = self._udp_request({"cmd": "get_state", "ts": time.time()}, expect_reply=True)
        if not resp or not resp.get("ok", False):
            raise RuntimeError("Failed to get remote state")
        state = resp.get("state", None)
        if state is None:
            raise RuntimeError("State missing in response")
        state = np.asarray(state, dtype=float).reshape(-1)
        if state.size != 2:
            raise RuntimeError(f"Invalid state shape: {state.shape}")
        return state