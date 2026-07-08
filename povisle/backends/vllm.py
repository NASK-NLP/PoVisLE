from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tqdm import tqdm

from povisle.backends.base import BaseBackend, GenerationResult
from povisle.backends.hf import build_processor_prompt, ensure_rgb_image
from povisle.configs import ModelConfig
from povisle.preprocess import preprocess


class VLLMBackend(BaseBackend):
    def __init__(self, config: ModelConfig) -> None:
        from transformers import AutoProcessor, AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.multimodal.utils import encode_image_url

        self._encode_image_url = encode_image_url

        model_args = dict(config.model_args)
        generation = config.generation
        self.use_chat = bool(model_args.pop("use_chat", False))
        trust_remote_code = model_args.get("trust_remote_code", True)

        try:
            self.processor = AutoProcessor.from_pretrained(
                config.model_id,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            self.processor = AutoTokenizer.from_pretrained(
                config.model_id,
                trust_remote_code=trust_remote_code,
            )

        self.chat_template_kwargs = generation.get("chat_template_kwargs", {})
        self.preprocessing_config = config.preprocessing
        self.sampling_params = SamplingParams(
            max_tokens=generation.get("max_new_tokens", 64),
            temperature=generation.get("temperature", 0.0),
            top_p=generation.get("top_p", 1.0),
            top_k=generation.get("top_k", 0),
            repetition_penalty=generation.get("repetition_penalty", 1.0),
            skip_special_tokens=generation.get("skip_special_tokens", True),
        )
        self.llm = LLM(model=config.model_id, **model_args)

    def generate(self, image: Any, prompt: str) -> GenerationResult:
        try:
            record = {"prompt": prompt, "image": image}

            if self.use_chat:
                outputs = self.llm.chat(
                    [self._build_chat_request(record)],
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                    chat_template_kwargs=self.chat_template_kwargs,
                )
            else:
                outputs = self.llm.generate(
                    [self._build_generation_request(record)],
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )

            return outputs[0].outputs[0].text.strip(), None
        except Exception as error:
            return None, str(error)

    def _build_chat_request(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        prompt = preprocess(record["prompt"], self.preprocessing_config)
        image = record["image"]

        if image is None:
            content: str | list[dict[str, Any]] = prompt
        else:
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self._encode_image_url(ensure_rgb_image(image)),
                    },
                },
            ]

        return [{"role": "user", "content": content}]

    def _build_generation_request(self, record: dict[str, Any]) -> dict[str, Any]:
        image = record["image"]

        request: dict[str, Any] = {
            "prompt": build_processor_prompt(
                self.processor,
                preprocess(record["prompt"], self.preprocessing_config),
                self.chat_template_kwargs,
                include_image=image is not None,
            ),
        }

        if image is not None:
            request["multi_modal_data"] = {"image": ensure_rgb_image(image)}

        return request

    def generate_batch(self, records: list[dict[str, Any]]) -> list[GenerationResult]:
        try:
            if self.use_chat:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    requests = list(
                        tqdm(
                            executor.map(self._build_chat_request, records),
                            total=len(records),
                            desc="Creating requests",
                        )
                    )

                outputs = self.llm.chat(
                    requests,
                    sampling_params=self.sampling_params,
                    use_tqdm=True,
                    chat_template_kwargs=self.chat_template_kwargs,
                )
            else:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    requests = list(
                        tqdm(
                            executor.map(self._build_generation_request, records),
                            total=len(records),
                            desc="Creating requests",
                        )
                    )

                outputs = self.llm.generate(
                    requests,
                    sampling_params=self.sampling_params,
                    use_tqdm=True,
                )

            return [(output.outputs[0].text.strip(), None) for output in outputs]
        except Exception as error:
            return [(None, str(error)) for _record in records]
