from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RigConfig:
    volume_root: Path
    models_dir: Path
    state_dir: Path
    log_dir: Path
    llama_server_bin: Path
    api_host: str
    api_port: int
    panel_host: str
    panel_port: int
    api_key: str
    panel_user: str
    panel_password: str
    hf_token: str
    allow_insecure: bool
    health_timeout: int
    stop_timeout: int

    @classmethod
    def from_env(cls) -> "RigConfig":
        project_dir = Path(__file__).resolve().parents[1]
        default_volume = (
            Path("/workspace") if Path("/workspace").is_dir() else project_dir / "data"
        )
        volume_root = (
            Path(os.environ.get("GGUF_VOLUME_ROOT", default_volume))
            .expanduser()
            .resolve()
        )
        state_dir = Path(
            os.environ.get("GGUF_STATE_DIR", volume_root / ".state" / "gguf-rig")
        )
        log_dir = Path(
            os.environ.get("GGUF_LOG_DIR", volume_root / "logs" / "gguf-rig")
        )

        configured_binary = os.environ.get("LLAMA_SERVER_BIN")
        if configured_binary:
            llama_server = Path(configured_binary).expanduser()
        else:
            found = shutil.which("llama-server")
            llama_server = (
                Path(found)
                if found
                else volume_root / "llama.cpp" / "build" / "bin" / "llama-server"
            )

        return cls(
            volume_root=volume_root,
            models_dir=Path(
                os.environ.get("GGUF_MODELS_DIR", volume_root / "models" / "gguf")
            ),
            state_dir=state_dir,
            log_dir=log_dir,
            llama_server_bin=llama_server,
            api_host=os.environ.get("GGUF_API_HOST", "0.0.0.0"),
            api_port=int(os.environ.get("GGUF_API_PORT", "8000")),
            panel_host=os.environ.get("GGUF_PANEL_HOST", "0.0.0.0"),
            panel_port=int(os.environ.get("GGUF_PANEL_PORT", "7860")),
            api_key=os.environ.get("GGUF_API_KEY", ""),
            panel_user=os.environ.get(
                "GGUF_PANEL_USER", os.environ.get("PANEL_USER", "")
            ),
            panel_password=os.environ.get(
                "GGUF_PANEL_PASSWORD", os.environ.get("PANEL_PASS", "")
            ),
            hf_token=os.environ.get("HF_TOKEN", ""),
            allow_insecure=_env_bool("GGUF_ALLOW_INSECURE"),
            health_timeout=int(os.environ.get("GGUF_HEALTH_TIMEOUT", "180")),
            stop_timeout=int(os.environ.get("GGUF_STOP_TIMEOUT", "15")),
        )

    @property
    def active_model_file(self) -> Path:
        return self.state_dir / "active_model.json"

    @property
    def pid_file(self) -> Path:
        return self.state_dir / "llama-server.pid"

    @property
    def server_log_file(self) -> Path:
        return self.log_dir / "llama-server.log"

    @property
    def local_api_url(self) -> str:
        return f"http://127.0.0.1:{self.api_port}"

    def ensure_directories(self) -> None:
        for path in (self.models_dir, self.state_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)

    def validate_security(self) -> None:
        if self.allow_insecure:
            return
        errors: list[str] = []
        if self.api_host not in {"127.0.0.1", "localhost", "::1"} and not self.api_key:
            errors.append(
                "GGUF_API_KEY is required when the API listens on a public interface"
            )
        if self.panel_host not in {"127.0.0.1", "localhost", "::1"}:
            if not self.panel_user or not self.panel_password:
                errors.append(
                    "GGUF_PANEL_USER and GGUF_PANEL_PASSWORD are required for a public panel"
                )
        if errors:
            raise RuntimeError(
                "; ".join(errors)
                + ". Set GGUF_ALLOW_INSECURE=1 only for trusted local development."
            )
