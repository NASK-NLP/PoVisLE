from importlib import metadata as importlib_metadata
import platform
import shutil
import subprocess
import sys
from typing import Any


def package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def collect_system_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            "povisle": package_version("povisle"),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "vllm": package_version("vllm"),
        },
    }

    if not shutil.which("nvidia-smi"):
        return metadata

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=5,
        )
    except Exception as error:
        metadata["gpus_error"] = str(error)
        return metadata

    gpus = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        gpus.append(
            {
                "name": parts[0],
                "memory_total": parts[1],
                "driver_version": parts[2],
            }
        )
    metadata["gpus"] = gpus
    return metadata
