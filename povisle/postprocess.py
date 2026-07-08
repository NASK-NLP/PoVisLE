import re
from typing import Callable

from povisle.configs import PostprocessingConfig


def postprocess(value: str | None, config: PostprocessingConfig | None) -> str | None:
    if value is None:
        return None

    result = value
    for step in config.steps if config else []:
        result = POSTPROCESSORS[step.name](result, **step.args)

    return result


def replace(value: str, pattern: str, replacement: str = "", flags: list[str] | None = None) -> str:
    return re.sub(pattern, replacement, value, flags=_regex_flags(flags))


def _regex_flags(flags: list[str] | None) -> int:
    enabled = 0
    for flag in flags or []:
        enabled |= REGEX_FLAGS[flag]
    return enabled


REGEX_FLAGS = {
    "ignorecase": re.IGNORECASE,
    "multiline": re.MULTILINE,
    "dotall": re.DOTALL,
}

POSTPROCESSORS: dict[str, Callable[..., str]] = {
    "replace": replace,
}
