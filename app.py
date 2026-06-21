#!/usr/bin/env python3
from __future__ import annotations

import html
import os
import re
import threading
from pathlib import Path

# Hugging Face reads cache paths at import time. Keep all cache metadata on the volume.
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_VOLUME = Path(
    os.environ.get(
        "GGUF_VOLUME_ROOT",
        "/workspace" if Path("/workspace").is_dir() else str(PROJECT_DIR / "data"),
    )
).expanduser()
os.environ.setdefault("HF_HOME", str(DEFAULT_VOLUME / ".hf"))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_VOLUME / ".hf" / "hub"))

import gradio as gr  # noqa: E402

from gguf_rig import (  # noqa: E402
    ActiveModel,
    LlamaServerManager,
    ModelLibrary,
    RemoteFile,
    RigConfig,
)
from gguf_rig.system import disk_stats, gpu_stats  # noqa: E402


CSS = """
:root { --rig-amber: #e8a33d; --rig-cyan: #4fd1c5; }
.gradio-container { max-width: 1180px !important; }
.rig-hero { padding: 18px 20px; border: 1px solid var(--border-color-primary); border-radius: 12px;
  background: radial-gradient(circle at 10% 0%, rgba(232,163,61,.14), transparent 44%),
              radial-gradient(circle at 95% 0%, rgba(79,209,197,.10), transparent 40%); }
.rig-hero h1 { margin: 0 0 4px; font-size: 1.7rem; }
.rig-hero p { margin: 0; opacity: .72; }
.rig-status { min-height: 164px; }
.rig-note { opacity: .78; font-size: .92rem; }
footer { display: none !important; }
"""


config = RigConfig.from_env()
config.ensure_directories()
library = ModelLibrary(config)
manager = LlamaServerManager(config, library)
remote_cache: dict[tuple[str, str], int | None] = {}
remote_groups: dict[tuple[str, str], list[RemoteFile]] = {}
remote_cache_lock = threading.Lock()
SHARD_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE
)


def _format_uptime(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def dashboard_markdown() -> str:
    status = manager.status()
    disk = disk_stats(config.models_dir)
    gpus = gpu_stats()
    state_icon = "🟢" if status["healthy"] else ("🟠" if status["running"] else "⚫")
    lines = [
        f"### {state_icon} llama-server: `{status['state']}`",
        f"- **Model:** `{status['model'] or 'none'}`",
        f"- **PID / uptime:** `{status['pid'] or '—'}` / `{_format_uptime(status['uptime_seconds'])}`",
        f"- **Volume:** `{disk['free_gib']:.1f} GiB free` / `{disk['total_gib']:.1f} GiB`",
        f"- **Local API:** `{status['api_url']}`",
    ]
    if gpus:
        for gpu in gpus:
            lines.append(
                f"- **GPU {gpu['index']}:** {gpu['name']} · {gpu['memory_used_mib']} / "
                f"{gpu['memory_total_mib']} MiB · {gpu['utilization']}% · {gpu['temperature']}°C"
            )
    else:
        lines.append("- **GPU:** `nvidia-smi unavailable`")
    return "\n".join(lines)


def _model_choices() -> list[tuple[str, str]]:
    return [
        (f"{record.id} · {record.quant} · {record.size_gib:.2f} GiB", record.id)
        for record in library.scan()
    ]


def refresh_library(selected: str | None = None):
    choices = _model_choices()
    values = {value for _, value in choices}
    value = selected if selected in values else (choices[0][1] if choices else None)
    return gr.update(choices=choices, value=value), dashboard_markdown()


def projector_choices(model_id: str | None):
    if not model_id:
        return gr.update(choices=[], value=None)
    try:
        model = library.get(model_id)
        root = config.models_dir.resolve()
        choices = [
            (path.relative_to(root).as_posix(), path.relative_to(root).as_posix())
            for path in sorted(model.path.parent.glob("*.gguf"))
            if "mmproj" in path.name.lower()
        ]
        default = (
            model.mmproj_path.relative_to(root).as_posix()
            if model.mmproj_path
            else None
        )
        return gr.update(choices=choices, value=default)
    except Exception:
        return gr.update(choices=[], value=None)


def list_remote(repo_id: str):
    try:
        files = library.remote_files(repo_id, token=config.hf_token or None)
        models = [item for item in files if not item.is_mmproj]
        projectors = [item for item in files if item.is_mmproj]
        visible_models: list[RemoteFile] = []
        with remote_cache_lock:
            for item in files:
                remote_cache[(repo_id.strip(), item.name)] = item.size_bytes
            for item in models:
                shard = SHARD_RE.match(item.name)
                if shard and int(shard.group("index")) != 1:
                    continue
                if shard:
                    prefix = shard.group("prefix")
                    total = int(shard.group("total"))
                    group = [
                        candidate
                        for candidate in models
                        if (candidate_shard := SHARD_RE.match(candidate.name))
                        and candidate_shard.group("prefix") == prefix
                        and int(candidate_shard.group("total")) == total
                    ]
                    group.sort(key=lambda candidate: candidate.name)
                else:
                    group = [item]
                remote_groups[(repo_id.strip(), item.name)] = group
                visible_models.append(item)

        model_choices = []
        for item in visible_models:
            group = remote_groups[(repo_id.strip(), item.name)]
            known_sizes = [
                candidate.size_bytes
                for candidate in group
                if candidate.size_bytes is not None
            ]
            total_size = sum(known_sizes) if len(known_sizes) == len(group) else None
            size_label = (
                "unknown" if total_size is None else f"{total_size / 1024**3:.2f} GiB"
            )
            shard_label = f" · {len(group)} shards" if len(group) > 1 else ""
            model_choices.append(
                (f"{item.name} · {item.quant}{shard_label} · {size_label}", item.name)
            )
        projector_options = [
            (f"{item.name} · {item.size_label}", item.name) for item in projectors
        ]
        summary = (
            f"Found **{len(models)} model file(s)** and **{len(projectors)} projector(s)**. "
            f"HF token: **{'configured' if config.hf_token else 'not configured'}**."
        )
        return (
            gr.update(
                choices=model_choices,
                value=model_choices[0][1] if model_choices else None,
            ),
            gr.update(choices=projector_options, value=None),
            summary,
        )
    except Exception as exc:
        return (
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            f"❌ {html.escape(str(exc))}",
        )


def download_remote(
    repo_id: str, filename: str | None, projector: str | None, progress=gr.Progress()
):
    if not filename:
        return "❌ Select a model file first.", gr.update()
    try:

        def report(value: float, description: str) -> None:
            progress(value, desc=description)

        with remote_cache_lock:
            group = remote_groups.get((repo_id.strip(), filename), [])
        if not group:
            group = [
                RemoteFile(
                    filename,
                    remote_cache.get((repo_id.strip(), filename)),
                    "unknown",
                    False,
                )
            ]
        known_sizes = [item.size_bytes for item in group if item.size_bytes is not None]
        if (
            len(known_sizes) == len(group)
            and library.free_bytes() < sum(known_sizes) + 1024**3
        ):
            raise OSError(
                "Not enough free volume space for all model shards and the 1 GiB safety margin"
            )

        downloaded: list[str] = []
        path = None
        for index, item in enumerate(group, start=1):
            progress(
                (index - 1) / max(1, len(group)), desc=f"Shard {index}/{len(group)}"
            )
            shard_path = library.download(
                repo_id,
                item.name,
                token=config.hf_token or None,
                expected_size=item.size_bytes,
                progress=report,
            )
            path = path or shard_path
            downloaded.append(str(shard_path))
        assert path is not None
        if projector:
            with remote_cache_lock:
                projector_size = remote_cache.get((repo_id.strip(), projector))
            downloaded.append(
                str(
                    library.download(
                        repo_id,
                        projector,
                        token=config.hf_token or None,
                        expected_size=projector_size,
                        progress=report,
                    )
                )
            )
        choices = _model_choices()
        model_id = path.relative_to(config.models_dir.resolve()).as_posix()
        return "✅ Downloaded:\n- " + "\n- ".join(downloaded), gr.update(
            choices=choices, value=model_id
        )
    except Exception as exc:
        return f"❌ {html.escape(str(exc))}", gr.update()


def activate_model(
    model_id: str | None,
    projector_id: str | None,
    context_size: float,
    gpu_layers: float,
    batch_size: float,
    chat_template: str,
    confirm_switch: bool,
):
    if not model_id:
        return "❌ Select a downloaded model.", dashboard_markdown()
    current = manager.status()
    if current["running"] and current["model"] != model_id and not confirm_switch:
        return (
            "⚠️ Check the switch confirmation box; active requests may be interrupted.",
            dashboard_markdown(),
        )
    active = ActiveModel(
        model_id=model_id,
        mmproj_id=projector_id or None,
        context_size=int(context_size),
        gpu_layers=int(gpu_layers),
        batch_size=int(batch_size),
        chat_template=(chat_template or "").strip(),
    )
    try:
        result = manager.switch(active)
        return f"✅ {result}", dashboard_markdown()
    except Exception as exc:
        return f"❌ {html.escape(str(exc))}", dashboard_markdown()


def restart_server():
    try:
        return f"✅ {manager.restart()}", dashboard_markdown()
    except Exception as exc:
        return f"❌ {html.escape(str(exc))}", dashboard_markdown()


def stop_server():
    try:
        return f"✅ {manager.stop()}", dashboard_markdown()
    except Exception as exc:
        return f"❌ {html.escape(str(exc))}", dashboard_markdown()


def build_app() -> gr.Blocks:
    choices = _model_choices()
    initial_model = choices[0][1] if choices else None
    with gr.Blocks(title="GGUF Inference Rig", css=CSS, theme=gr.themes.Base()) as demo:
        gr.HTML(
            "<div class='rig-hero'><h1>GGUF Inference Rig</h1>"
            "<p>Persistent GGUF library · llama-server · OpenAI-compatible API</p></div>"
        )
        with gr.Tabs():
            with gr.Tab("Dashboard"):
                dashboard = gr.Markdown(dashboard_markdown(), elem_classes="rig-status")
                with gr.Row():
                    refresh_dashboard = gr.Button("Refresh", variant="secondary")
                    restart_button = gr.Button("Restart server")
                    stop_button = gr.Button("Stop server", variant="stop")
                dashboard_message = gr.Markdown()

            with gr.Tab("Model Library"):
                with gr.Row():
                    with gr.Column(scale=3):
                        model_select = gr.Dropdown(
                            label="Models on persistent volume",
                            choices=choices,
                            value=initial_model,
                            filterable=True,
                        )
                    refresh_models = gr.Button("Rescan volume", scale=1)
                projector_select = gr.Dropdown(
                    label="Local mmproj (optional)", choices=[]
                )
                with gr.Accordion("Run configuration", open=True):
                    with gr.Row():
                        context_size = gr.Number(
                            label="Context size", value=8192, precision=0, minimum=512
                        )
                        gpu_layers = gr.Number(
                            label="GPU layers (-1 = all)",
                            value=-1,
                            precision=0,
                            minimum=-1,
                        )
                        batch_size = gr.Number(
                            label="Batch size", value=512, precision=0, minimum=1
                        )
                    chat_template = gr.Textbox(
                        label="Chat-template override (normally leave empty)",
                        lines=3,
                        placeholder="Use GGUF metadata unless this repository requires an override",
                    )
                confirm_switch = gr.Checkbox(
                    label="I understand that switching can interrupt in-flight requests"
                )
                activate_button = gr.Button("Activate model", variant="primary")
                activation_result = gr.Markdown()

                gr.Markdown("### Download from Hugging Face")
                gr.Markdown(
                    "The token is read only from the `HF_TOKEN` environment secret and is never entered or stored here.",
                    elem_classes="rig-note",
                )
                repo_id = gr.Textbox(
                    label="Repository", placeholder="organization/repository"
                )
                inspect_button = gr.Button("List GGUF files")
                remote_model = gr.Dropdown(
                    label="Quant / model file", choices=[], filterable=True
                )
                remote_projector = gr.Dropdown(
                    label="Matching mmproj (optional)", choices=[], filterable=True
                )
                remote_summary = gr.Markdown()
                download_button = gr.Button(
                    "Download to persistent volume", variant="primary"
                )
                download_result = gr.Markdown()

            with gr.Tab("Console"):
                console = gr.Textbox(
                    label="llama-server output",
                    value=manager.logs(),
                    lines=28,
                    interactive=False,
                )
                console_refresh = gr.Button("Refresh log")

            with gr.Tab("Settings"):
                gr.Markdown(
                    f"""
### Runtime

- Model volume: `{config.models_dir}`
- State: `{config.state_dir}`
- llama-server: `{config.llama_server_bin}`
- API listener: `{config.api_host}:{config.api_port}`
- API key: **{"configured" if config.api_key else "missing"}**
- Panel authentication: **{"configured" if config.panel_user and config.panel_password else "missing"}**
- Hugging Face token: **{"configured" if config.hf_token else "not configured"}**

Secrets are intentionally environment-only. Change them in RunPod Secrets/Environment Variables and restart the pod.
                    """
                )

        refresh_dashboard.click(dashboard_markdown, outputs=dashboard)
        restart_button.click(restart_server, outputs=[dashboard_message, dashboard])
        stop_button.click(stop_server, outputs=[dashboard_message, dashboard])
        refresh_models.click(
            refresh_library, inputs=model_select, outputs=[model_select, dashboard]
        )
        model_select.change(
            projector_choices, inputs=model_select, outputs=projector_select
        )
        activate_button.click(
            activate_model,
            inputs=[
                model_select,
                projector_select,
                context_size,
                gpu_layers,
                batch_size,
                chat_template,
                confirm_switch,
            ],
            outputs=[activation_result, dashboard],
            concurrency_limit=1,
        )
        inspect_button.click(
            list_remote,
            inputs=repo_id,
            outputs=[remote_model, remote_projector, remote_summary],
        )
        download_button.click(
            download_remote,
            inputs=[repo_id, remote_model, remote_projector],
            outputs=[download_result, model_select],
            concurrency_limit=1,
        )
        console_refresh.click(lambda: manager.logs(), outputs=console)
        timer = gr.Timer(5)
        timer.tick(dashboard_markdown, outputs=dashboard)
        timer.tick(lambda: manager.logs(), outputs=console)

    return demo


def _restore_in_background() -> None:
    try:
        manager.restore()
    except Exception as exc:
        manager._append_log(f"Automatic restore failed: {exc}")


def main() -> None:
    config.validate_security()
    demo = build_app()
    threading.Thread(target=_restore_in_background, daemon=True).start()
    auth: tuple[str, str] | None = None
    if config.panel_user and config.panel_password:
        auth = (config.panel_user, config.panel_password)
    demo.queue(default_concurrency_limit=4).launch(
        server_name=config.panel_host,
        server_port=config.panel_port,
        auth=auth,
        show_error=True,
    )


if __name__ == "__main__":
    main()
