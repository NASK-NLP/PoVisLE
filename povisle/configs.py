from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class PostprocessingStep:
    name: str
    args: dict[str, Any]

    @classmethod
    def from_value(cls, value: str | dict[str, Any]) -> "PostprocessingStep":
        if isinstance(value, str):
            return cls(name=value, args={})
        return cls(name=value["name"], args=value.get("args", {}))


@dataclass
class PostprocessingConfig:
    steps: list[PostprocessingStep]

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PostprocessingConfig":
        return cls(steps=[PostprocessingStep.from_value(step) for step in (data or {}).get("steps", [])])


@dataclass
class PreprocessingConfig:
    steps: list[PostprocessingStep]

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PreprocessingConfig":
        return cls(steps=[PostprocessingStep.from_value(step) for step in (data or {}).get("steps", [])])


@dataclass
class ModelConfig:
    model_id: str
    org: str | None
    name: str | None
    model_family: str | None
    model_type: str | None
    model_size: str | None
    backend: Literal["vllm", "hf", "api", "random"]
    model_args: dict[str, Any]
    generation: dict[str, Any]
    preprocessing: PreprocessingConfig
    postprocessing: PostprocessingConfig

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelConfig":
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        return cls(
            model_id=data["model_id"],
            org=data.get("org"),
            name=data.get("name"),
            model_family=data.get("model_family"),
            model_type=data.get("model_type"),
            model_size=data.get("model_size"),
            backend=data["backend"],
            model_args=data.get("model_args", {}),
            generation=data.get("generation", {}),
            preprocessing=PreprocessingConfig.from_dict(data.get("preprocessing")),
            postprocessing=PostprocessingConfig.from_dict(data.get("postprocessing")),
        )
