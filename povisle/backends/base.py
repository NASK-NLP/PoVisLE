from typing import Any

GenerationResult = tuple[str | None, str | None]


class BaseBackend:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def generate(self, image: Any, prompt: str) -> GenerationResult:
        raise NotImplementedError

    def generate_batch(self, records: list[dict[str, Any]]) -> list[GenerationResult]:
        return [self.generate(record["image"], record["prompt"]) for record in records]

    def close(self) -> None:
        return None
