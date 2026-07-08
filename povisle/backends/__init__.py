from povisle.backends.base import BaseBackend
from povisle.backends.api import APIBackend
from povisle.backends.hf import HuggingFaceBackend
from povisle.backends.random import RandomBackend
from povisle.backends.vllm import VLLMBackend
from povisle.configs import ModelConfig


def create_backend(config: ModelConfig) -> BaseBackend:
    if config.backend == "api":
        return APIBackend(config)
    if config.backend == "hf":
        return HuggingFaceBackend(config)
    if config.backend == "random":
        return RandomBackend(config)
    if config.backend == "vllm":
        return VLLMBackend(config)
    raise ValueError(f"Unsupported backend: {config.backend}")
