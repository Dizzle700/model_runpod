from __future__ import annotations

import json
from pathlib import Path

import pytest

from gguf_rig.config import RigConfig
from gguf_rig.library import ModelLibrary, detect_quant
from gguf_rig.process_manager import ActiveModel, LlamaServerManager


def make_config(tmp_path: Path, **overrides) -> RigConfig:
    values = dict(
        volume_root=tmp_path,
        models_dir=tmp_path / "models",
        state_dir=tmp_path / "state",
        log_dir=tmp_path / "logs",
        llama_server_bin=tmp_path / "llama-server",
        api_host="127.0.0.1",
        api_port=8000,
        panel_host="127.0.0.1",
        panel_port=7860,
        api_key="",
        panel_user="",
        panel_password="",
        hf_token="",
        allow_insecure=False,
        health_timeout=1,
        stop_timeout=1,
    )
    values.update(overrides)
    return RigConfig(**values)


def test_quant_detection():
    assert detect_quant("model-Q4_K_M.gguf") == "Q4_K_M"
    assert detect_quant("model.IQ3_XXS.gguf") == "IQ3_XXS"
    assert detect_quant("model-f16.gguf") == "F16"
    assert detect_quant("model.gguf") == "unknown"


def test_library_scan_pairs_single_projector(tmp_path: Path):
    config = make_config(tmp_path)
    repo = config.models_dir / "org" / "repo"
    repo.mkdir(parents=True)
    model = repo / "demo-Q5_K_M.gguf"
    model.write_bytes(b"model")
    projector = repo / "mmproj-demo-f16.gguf"
    projector.write_bytes(b"projector")

    records = ModelLibrary(config).scan()

    assert [record.id for record in records] == ["org/repo/demo-Q5_K_M.gguf"]
    assert records[0].quant == "Q5_K_M"
    assert records[0].mmproj_path == projector


def test_library_scan_collapses_shards_and_sums_size(tmp_path: Path):
    config = make_config(tmp_path)
    repo = config.models_dir / "org" / "repo"
    repo.mkdir(parents=True)
    first = repo / "large-Q4_K_M-00001-of-00002.gguf"
    first.write_bytes(b"123")
    (repo / "large-Q4_K_M-00002-of-00002.gguf").write_bytes(b"4567")

    records = ModelLibrary(config).scan()

    assert len(records) == 1
    assert records[0].path == first
    assert records[0].size_bytes == 7


def test_library_rejects_path_escape(tmp_path: Path):
    library = ModelLibrary(make_config(tmp_path))
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"x")

    with pytest.raises(ValueError, match="escapes"):
        library.get("../outside.gguf")


def test_remote_validation():
    assert ModelLibrary.validate_repo_id("org/repo") == "org/repo"
    assert (
        ModelLibrary.validate_remote_name("sub/model-Q4_K_M.gguf")
        == "sub/model-Q4_K_M.gguf"
    )
    with pytest.raises(ValueError):
        ModelLibrary.validate_repo_id("not-a-repo")
    with pytest.raises(ValueError):
        ModelLibrary.validate_remote_name("../secret.gguf")


def test_public_listeners_require_secrets(tmp_path: Path):
    config = make_config(tmp_path, api_host="0.0.0.0", panel_host="0.0.0.0")
    with pytest.raises(RuntimeError, match="GGUF_API_KEY"):
        config.validate_security()


def test_build_command_masks_no_configuration_and_uses_projector(tmp_path: Path):
    config = make_config(tmp_path, api_key="secret")
    config.llama_server_bin.write_bytes(b"binary")
    repo = config.models_dir / "org" / "repo"
    repo.mkdir(parents=True)
    (repo / "demo-Q4_K_M.gguf").write_bytes(b"model")
    (repo / "mmproj-demo-f16.gguf").write_bytes(b"projector")
    manager = LlamaServerManager(config, ModelLibrary(config))

    command = manager.build_command(ActiveModel(model_id="org/repo/demo-Q4_K_M.gguf"))

    assert command[0] == str(config.llama_server_bin)
    assert command[command.index("--api-key") + 1] == "secret"
    assert command[command.index("--mmproj") + 1].endswith("mmproj-demo-f16.gguf")


def test_saved_state_round_trip(tmp_path: Path):
    config = make_config(tmp_path)
    repo = config.models_dir / "org" / "repo"
    repo.mkdir(parents=True)
    (repo / "demo-Q8_0.gguf").write_bytes(b"model")
    manager = LlamaServerManager(config, ModelLibrary(config))
    active = ActiveModel(
        model_id="org/repo/demo-Q8_0.gguf", context_size=4096, batch_size=128
    )

    manager._write_state(active)

    assert manager.load_saved() == active
    payload = json.loads(config.active_model_file.read_text())
    assert payload["schema_version"] == 1
    assert "api_key" not in payload
    assert "hf_token" not in payload
