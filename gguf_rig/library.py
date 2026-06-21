from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable

from .config import RigConfig


QUANT_RE = re.compile(
    r"(?:^|[-_.])((?:IQ|Q|TQ)\d(?:_[A-Z0-9]+)*|F(?:16|32)|BF16)(?:[-_.]|$)",
    re.IGNORECASE,
)
REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
SHARD_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE
)


@dataclass(frozen=True)
class ModelRecord:
    id: str
    path: Path
    size_bytes: int
    quant: str
    mmproj_path: Path | None = None

    @property
    def size_gib(self) -> float:
        return self.size_bytes / 1024**3


@dataclass(frozen=True)
class RemoteFile:
    name: str
    size_bytes: int | None
    quant: str
    is_mmproj: bool

    @property
    def size_label(self) -> str:
        return (
            "unknown"
            if self.size_bytes is None
            else f"{self.size_bytes / 1024**3:.2f} GiB"
        )


def detect_quant(filename: str) -> str:
    match = QUANT_RE.search(Path(filename).name)
    return match.group(1).upper() if match else "unknown"


def _is_mmproj(name: str) -> bool:
    return "mmproj" in Path(name).name.lower()


class ModelLibrary:
    def __init__(self, config: RigConfig):
        self.config = config
        self.config.ensure_directories()

    def _safe_local_path(self, path: Path | str) -> Path:
        root = self.config.models_dir.resolve()
        candidate = Path(path).expanduser().resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"Path escapes the model library: {candidate}")
        return candidate

    @staticmethod
    def validate_repo_id(repo_id: str) -> str:
        repo_id = repo_id.strip()
        if not REPO_RE.fullmatch(repo_id):
            raise ValueError("Repository must look like organization/name")
        return repo_id

    @staticmethod
    def validate_remote_name(filename: str) -> str:
        filename = filename.strip()
        pure = PurePosixPath(filename)
        if (
            not filename
            or pure.is_absolute()
            or ".." in pure.parts
            or not filename.lower().endswith(".gguf")
        ):
            raise ValueError("Unsafe or non-GGUF filename")
        return filename

    def scan(self) -> list[ModelRecord]:
        root = self.config.models_dir.resolve()
        projectors_by_dir: dict[Path, list[Path]] = {}
        model_paths: list[Path] = []
        for path in root.rglob("*.gguf"):
            if not path.is_file():
                continue
            if _is_mmproj(path.name):
                projectors_by_dir.setdefault(path.parent, []).append(path)
            else:
                model_paths.append(path)

        records: list[ModelRecord] = []
        for path in sorted(model_paths, key=lambda item: str(item).lower()):
            shard = SHARD_RE.match(path.name)
            if shard and int(shard.group("index")) != 1:
                continue
            projectors = sorted(projectors_by_dir.get(path.parent, []))
            size_bytes = path.stat().st_size
            if shard:
                siblings = path.parent.glob(
                    f"{shard.group('prefix')}-*-of-{shard.group('total')}.gguf"
                )
                size_bytes = sum(
                    item.stat().st_size for item in siblings if item.is_file()
                )
            records.append(
                ModelRecord(
                    id=path.relative_to(root).as_posix(),
                    path=path,
                    size_bytes=size_bytes,
                    quant=detect_quant(path.name),
                    mmproj_path=projectors[0] if len(projectors) == 1 else None,
                )
            )
        return records

    def get(self, model_id: str) -> ModelRecord:
        path = self._safe_local_path(self.config.models_dir / model_id)
        if (
            not path.is_file()
            or path.suffix.lower() != ".gguf"
            or _is_mmproj(path.name)
        ):
            raise FileNotFoundError(f"Model not found: {model_id}")
        return next(
            (record for record in self.scan() if record.path == path),
            ModelRecord(
                id=path.relative_to(self.config.models_dir.resolve()).as_posix(),
                path=path,
                size_bytes=path.stat().st_size,
                quant=detect_quant(path.name),
            ),
        )

    def resolve_projector(
        self, model: ModelRecord, projector_name: str | None
    ) -> Path | None:
        if not projector_name:
            return model.mmproj_path
        projector = self._safe_local_path(model.path.parent / projector_name)
        if not projector.is_file() or not _is_mmproj(projector.name):
            raise FileNotFoundError(f"Projector not found: {projector_name}")
        return projector

    def remote_files(self, repo_id: str, token: str | None = None) -> list[RemoteFile]:
        repo_id = self.validate_repo_id(repo_id)
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError("huggingface-hub is not installed") from exc

        info = HfApi(token=token or None).model_info(
            repo_id=repo_id, files_metadata=True
        )
        files: list[RemoteFile] = []
        for sibling in info.siblings or []:
            name = getattr(sibling, "rfilename", "")
            if not name.lower().endswith(".gguf"):
                continue
            files.append(
                RemoteFile(
                    name=name,
                    size_bytes=getattr(sibling, "size", None),
                    quant=detect_quant(name),
                    is_mmproj=_is_mmproj(name),
                )
            )
        return sorted(files, key=lambda item: (item.is_mmproj, item.name.lower()))

    def free_bytes(self) -> int:
        return shutil.disk_usage(self.config.models_dir).free

    def download(
        self,
        repo_id: str,
        filename: str,
        *,
        token: str | None = None,
        expected_size: int | None = None,
        progress: Callable[[float, str], None] | None = None,
    ) -> Path:
        repo_id = self.validate_repo_id(repo_id)
        filename = self.validate_remote_name(filename)
        if expected_size is not None and self.free_bytes() < expected_size + 1024**3:
            raise OSError(
                "Not enough free volume space (the download requires a 1 GiB safety margin)"
            )

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("huggingface-hub is not installed") from exc

        org, repo = repo_id.split("/", 1)
        destination = self.config.models_dir / org / repo
        destination.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(0.05, f"Downloading {filename}")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=destination,
            token=token or None,
        )
        result = self._safe_local_path(Path(downloaded))
        if progress:
            progress(1.0, f"Saved to {result}")
        return result

    def delete(self, model_ids: Iterable[str]) -> int:
        removed = 0
        for model_id in model_ids:
            model = self.get(model_id)
            model.path.unlink()
            removed += 1
        return removed
