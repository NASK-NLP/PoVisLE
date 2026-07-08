import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset

from povisle.configs import ModelConfig


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(to_jsonable(record), ensure_ascii=False))
            handle.write("\n")


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    folded = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = folded.lower().strip()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", normalize_text(text))
    return re.sub(r"_+", "_", slug).strip("._-") or "run"


def circular_rotations[T](values: tuple[T, ...]) -> list[tuple[T, ...]]:
    return [values[index:] + values[:index] for index in range(len(values))]


def format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def make_run_id(config: ModelConfig) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_label = config.model_id
    if config.org and config.name:
        model_label = f"{config.org}/{config.name}"
    elif config.name:
        model_label = config.name
    return f"{timestamp}_{config.backend}_{slugify(model_label)}"


def load_task_dataset(
    dataset_id: str,
    subset: str,
    split: str,
    revision: str | None = None,
    limit: int | None = None,
):
    dataset = load_dataset(dataset_id, subset, split=split, revision=revision)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value


def env_value(name: str | None) -> str | None:
    return os.environ.get(name) if name else None
