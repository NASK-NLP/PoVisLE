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
from povisle.preprocess import preprocess
from povisle.utils import env_value


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
            prompt = preprocess(prompt, self.preprocessing_config)
            content = [{"type": "text", "text": prompt}]
            if image is not None:
                content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image)}})
            message = {
                "role": "user",
                "content": content,
            }
            response = self._invoke_with_retry([message])
            return response.content.strip(), None
        except Exception as error:
            return None, str(error)

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

        return invoke().choices[0].message


def image_to_data_url(image: Any) -> str:
    buffer = BytesIO()
    ensure_rgb_image(image).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
