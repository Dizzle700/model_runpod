from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import RigConfig
from .library import ModelLibrary


@dataclass(frozen=True)
class ActiveModel:
    model_id: str
    mmproj_id: str | None = None
    context_size: int = 8192
    gpu_layers: int = -1
    batch_size: int = 512
    chat_template: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActiveModel":
        return cls(
            model_id=str(data["model_id"]),
            mmproj_id=data.get("mmproj_id") or None,
            context_size=int(data.get("context_size", 8192)),
            gpu_layers=int(data.get("gpu_layers", -1)),
            batch_size=int(data.get("batch_size", 512)),
            chat_template=str(data.get("chat_template", "")),
        )

    def validate(self) -> None:
        if not 512 <= self.context_size <= 1_048_576:
            raise ValueError("Context size must be between 512 and 1,048,576")
        if not -1 <= self.gpu_layers <= 10_000:
            raise ValueError("GPU layers must be -1 (all) or a non-negative number")
        if not 1 <= self.batch_size <= 65_536:
            raise ValueError("Batch size must be between 1 and 65,536")
        if "\x00" in self.chat_template:
            raise ValueError("Chat template contains an invalid NUL character")


class LlamaServerManager:
    """Owns exactly one llama-server process and its persisted run configuration."""

    def __init__(self, config: RigConfig, library: ModelLibrary):
        self.config = config
        self.library = library
        self.config.ensure_directories()
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._active: ActiveModel | None = None
        self._started_at: float | None = None
        self._log_lines: deque[str] = deque(maxlen=2_000)
        self._log_handle = None
        self._intentional_stop = False
        self._cleanup_stale_process()
        atexit.register(self.shutdown)

    def _append_log(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._log_lines.append(f"[{stamp}] {message.rstrip()}")

    def logs(self, limit: int = 300) -> str:
        with self._lock:
            return "\n".join(list(self._log_lines)[-max(1, int(limit)) :])

    def _cleanup_stale_process(self) -> None:
        if not self.config.pid_file.exists():
            return
        try:
            pid = int(self.config.pid_file.read_text(encoding="utf-8").strip())
            cmdline_path = Path(f"/proc/{pid}/cmdline")
            cmdline = (
                cmdline_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
            )
            expected_name = self.config.llama_server_bin.name
            owns_process = (
                expected_name in cmdline
                and str(self.config.models_dir) in cmdline
                and f"--port {self.config.api_port}" in cmdline
            )
            if owns_process:
                self._append_log(f"Stopping stale managed llama-server process {pid}")
                try:
                    os.killpg(pid, signal.SIGTERM)
                    deadline = time.monotonic() + self.config.stop_timeout
                    while time.monotonic() < deadline:
                        try:
                            os.kill(pid, 0)
                        except ProcessLookupError:
                            break
                        time.sleep(0.1)
                    else:
                        self._append_log(
                            f"Stale process {pid} ignored SIGTERM; sending SIGKILL"
                        )
                        os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        except (OSError, ValueError):
            pass
        finally:
            self.config.pid_file.unlink(missing_ok=True)

    def _model_paths(self, active: ActiveModel) -> tuple[Path, Path | None]:
        active.validate()
        model = self.library.get(active.model_id)
        projector = None
        if active.mmproj_id:
            candidate = self.config.models_dir / active.mmproj_id
            projector = self.library._safe_local_path(candidate)
            if not projector.is_file() or "mmproj" not in projector.name.lower():
                raise FileNotFoundError(f"Projector not found: {active.mmproj_id}")
        elif model.mmproj_path:
            projector = model.mmproj_path
        return model.path, projector

    def build_command(self, active: ActiveModel) -> list[str]:
        model_path, projector = self._model_paths(active)
        command = [
            str(self.config.llama_server_bin),
            "--model",
            str(model_path),
            "--host",
            self.config.api_host,
            "--port",
            str(self.config.api_port),
            "--ctx-size",
            str(active.context_size),
            "--n-gpu-layers",
            str(active.gpu_layers),
            "--batch-size",
            str(active.batch_size),
        ]
        if self.config.api_key:
            command.extend(["--api-key", self.config.api_key])
        if projector:
            command.extend(["--mmproj", str(projector)])
        if active.chat_template:
            command.extend(["--chat-template", active.chat_template])
        return command

    def _health(self, timeout: float = 1.5) -> tuple[bool, str]:
        request = urllib.request.Request(f"{self.config.local_api_url}/health")
        if self.config.api_key:
            request.add_header("Authorization", f"Bearer {self.config.api_key}")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read(2048).decode(errors="replace")
                return 200 <= response.status < 300, body
        except urllib.error.HTTPError as exc:
            return False, f"HTTP {exc.code}"
        except (OSError, TimeoutError) as exc:
            return False, str(exc)

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.config.health_timeout
        last_error = "not ready"
        while time.monotonic() < deadline:
            process = self._process
            if process is None:
                raise RuntimeError("llama-server process disappeared")
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"llama-server exited during startup with code {return_code}\n{self.logs(40)}"
                )
            healthy, detail = self._health()
            if healthy:
                return
            last_error = detail
            time.sleep(1)
        raise TimeoutError(
            f"llama-server did not become ready in {self.config.health_timeout}s: {last_error}"
        )

    def _write_state(self, active: ActiveModel) -> None:
        payload = {
            "schema_version": 1,
            **asdict(active),
            "updated_at": int(time.time()),
        }
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix="active-model-", suffix=".json", dir=self.config.state_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.config.active_model_file)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def load_saved(self) -> ActiveModel | None:
        try:
            data = json.loads(self.config.active_model_file.read_text(encoding="utf-8"))
            active = ActiveModel.from_dict(data)
            self._model_paths(active)
            return active
        except FileNotFoundError:
            return None
        except (ValueError, TypeError, json.JSONDecodeError, OSError) as exc:
            self._append_log(f"Ignoring invalid saved state: {exc}")
            return None

    def _launch(self, active: ActiveModel, persist: bool) -> None:
        if not self.config.llama_server_bin.is_file():
            raise FileNotFoundError(
                f"llama-server binary not found at {self.config.llama_server_bin}. Run install_runpod.sh first."
            )
        command = self.build_command(active)
        safe_command = [
            "***" if index and command[index - 1] == "--api-key" else part
            for index, part in enumerate(command)
        ]
        self._append_log("Starting: " + " ".join(safe_command))
        self.config.server_log_file.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.config.server_log_file.open(
            "a", encoding="utf-8", buffering=1
        )
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self.config.pid_file.write_text(f"{self._process.pid}\n", encoding="utf-8")
        self._active = active
        self._started_at = time.time()
        threading.Thread(
            target=self._capture_output, args=(self._process,), daemon=True
        ).start()
        try:
            self._wait_ready()
        except Exception:
            self._stop_locked()
            raise
        self._append_log(f"Ready on {self.config.local_api_url}")
        if persist:
            self._write_state(active)

    def _capture_output(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return
        try:
            for line in process.stdout:
                clean = line.rstrip()
                if clean:
                    self._append_log(clean)
                    try:
                        if self._log_handle and not self._log_handle.closed:
                            self._log_handle.write(clean + "\n")
                    except (OSError, ValueError):
                        pass
        finally:
            return_code = process.wait()
            if process is self._process and not self._intentional_stop:
                self._append_log(
                    f"llama-server exited unexpectedly with code {return_code}"
                )

    def start(self, active: ActiveModel, *, persist: bool = True) -> None:
        with self._lock:
            if self._process and self._process.poll() is None:
                raise RuntimeError(
                    "llama-server is already running; use switch() or restart()"
                )
            self._intentional_stop = False
            self._launch(active, persist)

    def switch(self, active: ActiveModel) -> str:
        with self._lock:
            previous = (
                self._active if self._process and self._process.poll() is None else None
            )
            if previous == active:
                return "The selected model is already active."
            self._stop_locked()
            try:
                self._intentional_stop = False
                self._launch(active, persist=True)
                return f"Activated {active.model_id}"
            except Exception as new_error:
                self._append_log(f"Activation failed: {new_error}")
                if previous:
                    self._append_log(f"Rolling back to {previous.model_id}")
                    try:
                        self._intentional_stop = False
                        self._launch(previous, persist=False)
                    except Exception as rollback_error:
                        raise RuntimeError(
                            f"New model failed: {new_error}; rollback also failed: {rollback_error}"
                        ) from new_error
                    raise RuntimeError(
                        f"New model failed; rolled back to {previous.model_id}: {new_error}"
                    ) from new_error
                raise

    def _stop_locked(self) -> None:
        process = self._process
        if process is None:
            self.config.pid_file.unlink(missing_ok=True)
            return
        self._intentional_stop = True
        if process.poll() is None:
            self._append_log(f"Sending SIGTERM to process group {process.pid}")
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=self.config.stop_timeout)
            except subprocess.TimeoutExpired:
                self._append_log("Grace period expired; sending SIGKILL")
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
            except ProcessLookupError:
                pass
        self._process = None
        self._started_at = None
        self.config.pid_file.unlink(missing_ok=True)
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

    def stop(self) -> str:
        with self._lock:
            if not self._process or self._process.poll() is not None:
                self._process = None
                return "llama-server is already stopped."
            self._stop_locked()
            return "llama-server stopped."

    def restart(self) -> str:
        with self._lock:
            active = self._active or self.load_saved()
            if not active:
                raise RuntimeError("No active or saved model to restart")
            self._stop_locked()
            self._intentional_stop = False
            self._launch(active, persist=True)
            return f"Restarted {active.model_id}"

    def restore(self) -> str:
        active = self.load_saved()
        if not active:
            return "No saved model to restore."
        self.start(active, persist=False)
        return f"Restored {active.model_id}"

    def status(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            running = bool(process and process.poll() is None)
            healthy, health_detail = self._health() if running else (False, "stopped")
            return {
                "state": "ready"
                if healthy
                else ("starting/unhealthy" if running else "stopped"),
                "running": running,
                "healthy": healthy,
                "health_detail": health_detail,
                "pid": process.pid if running and process else None,
                "model": self._active.model_id if self._active else None,
                "uptime_seconds": int(time.time() - self._started_at)
                if running and self._started_at
                else 0,
                "api_url": self.config.local_api_url,
            }

    def shutdown(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
