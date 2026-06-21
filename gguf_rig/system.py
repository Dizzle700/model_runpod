from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any


def gpu_stats() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    devices = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        index, name, temperature, utilization, used, total = parts
        devices.append(
            {
                "index": int(index),
                "name": name,
                "temperature": int(temperature),
                "utilization": int(utilization),
                "memory_used_mib": int(used),
                "memory_total_mib": int(total),
            }
        )
    return devices


def disk_stats(path: Path) -> dict[str, float]:
    usage = shutil.disk_usage(path)
    return {
        "free_gib": usage.free / 1024**3,
        "used_gib": usage.used / 1024**3,
        "total_gib": usage.total / 1024**3,
    }
