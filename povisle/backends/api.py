import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any

import openai
import stamina
from openai import OpenAI
from tqdm import tqdm

from povisle.backends.base import BaseBackend, GenerationResult
from povisle.backends.hf import ensure_rgb_image
from povisle.configs import ModelConfig
from povisle.logger import get_logger
from povisle.preprocess import preprocess
from povisle.utils import env_value

logger = get_logger(__name__)


class APIBackend(BaseBackend):
    def __init__(self, config: ModelConfig) -> None:
        model_args = config.model_args
        self.model_id = config.model_id
        self.generation = config.generation
        self.preprocessing_config = config.preprocessing
        self.num_threads = int(model_args.get("num_threads", 8))
        self.max_retries = int(model_args.get("max_retries", 5))
        self.client = OpenAI(
            api_key=env_value(model_args.get("api_key_env", "OPENAI_API_KEY")),
            base_url=model_args.get("base_url"),
            timeout=model_args.get("timeout"),
        )

    def generate(self, image: Any, prompt: str) -> GenerationResult:
        try:
            messages = build_chat_messages(image, prompt, self.preprocessing_config)
            response = self._invoke_with_retry(messages)
            metadata = response_metadata(response)
            if not response.choices:
                error = f"API response has no choices: {metadata}"
                logger.warning("Error during generation for prompt: %s. Error: %s", prompt, error)
                return None, error, metadata
            message = response.choices[0].message
            if message.content is None:
                error = f"API response message has no text content: {metadata}"
                logger.warning("Error during generation for prompt: %s. Error: %s", prompt, error)
                return None, error, metadata
            return message.content.strip(), None, metadata
        except Exception as error:
            logger.warning("Error during generation for prompt: %s. Error: %s", prompt, error)
            return None, str(error), None

    def generate_batch(self, records: list[dict[str, Any]]) -> list[GenerationResult]:
        results: list[GenerationResult | None] = [None] * len(records)
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {
                executor.submit(self.generate, record["image"], record["prompt"]): index
                for index, record in enumerate(records)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Invoking API"):
                results[futures[future]] = future.result()

        return [result for result in results if result is not None]

    def _invoke_with_retry(self, messages: list[dict[str, Any]]) -> Any:
        @stamina.retry(
            on=(
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.RateLimitError,
                openai.InternalServerError,
                TimeoutError,
                ConnectionError,
            ),
            attempts=self.max_retries,
        )
        def invoke() -> Any:
            return self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                **self.generation,
            )

        return invoke()


def build_chat_messages(
    image: Any,
    prompt: str,
    preprocessing_config: Any,
    image_url: str | None = None,
) -> list[dict[str, Any]]:
    prompt = preprocess(prompt, preprocessing_config)
    content = [{"type": "text", "text": prompt}]
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    elif image is not None:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image)}})
    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def generation_body(generation: dict[str, Any]) -> dict[str, Any]:
    body = dict(generation)
    extra_body = body.pop("extra_body", None)
    body.pop("extra_headers", None)
    if extra_body:
        body.update(extra_body)
    return body


def response_metadata(response: Any) -> dict[str, Any]:
    metadata = {
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
        "created": getattr(response, "created", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
    }

    usage = getattr(response, "usage", None)
    if usage is not None:
        metadata["usage"] = dump_model(usage)

    choices = getattr(response, "choices", None)
    if choices:
        metadata["finish_reason"] = getattr(choices[0], "finish_reason", None)

    return {key: value for key, value in metadata.items() if value is not None}


def dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return value


def image_to_data_url(image: Any) -> str:
    buffer = BytesIO()
    ensure_rgb_image(image).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
