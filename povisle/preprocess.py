from typing import Callable

from povisle.configs import PreprocessingConfig


def preprocess(value: str, config: PreprocessingConfig | None) -> str:
    result = value
    for step in config.steps if config else []:
        result = PREPROCESSORS[step.name](result, **step.args)
    return result


def prepend(value: str, prefix: str, if_missing: bool = True) -> str:
    if if_missing and value.startswith(prefix):
        return value
    return f"{prefix}{value}"


PREPROCESSORS: dict[str, Callable[..., str]] = {
    "prepend": prepend,
}
