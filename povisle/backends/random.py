from __future__ import annotations

import random
from typing import Any

from povisle.backends.base import BaseBackend, GenerationResult
from povisle.configs import ModelConfig
from povisle.tasks.mcq import LABELS as MCQ_LABELS


class RandomBackend(BaseBackend):
    def __init__(self, config: ModelConfig) -> None:
        seed = config.model_args.get("seed", config.generation.get("seed"))
        self.random = random.Random(seed)

    def generate(self, image: Any, prompt: str) -> GenerationResult:
        raise NotImplementedError("Use generate_batch for RandomBackend")

    def generate_batch(self, records: list[dict[str, Any]]) -> list[GenerationResult]:
        return [self.generate_record(record) for record in records]

    def generate_record(self, record: dict[str, Any]) -> GenerationResult:
        task = record.get("task")
        if task == "mcq":
            row = record.get("row")
            labels = [
                label
                for label in MCQ_LABELS
                if isinstance(row, dict) and isinstance(row.get(label), str) and row[label].strip()
            ]
            if not labels:
                return "", None, None
            return self.random.choice(labels), None, None
        if task == "yn":
            return self.random.choice(("yes", "no")), None, None
        if task == "open":
            return "", None, None

        raise ValueError(f"Unsupported task: {task}")
