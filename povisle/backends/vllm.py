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
        is_mistral_tokenizer = model_args.get("tokenizer_mode") == "mistral"
        self.generate_batch_size = int(model_args.pop("generate_batch_size", 1 if is_mistral_tokenizer else 0))
        tokenizer_kwargs = dict(model_args.pop("tokenizer_kwargs", {}))
        if is_mistral_tokenizer:
            tokenizer_kwargs.setdefault("fix_mistral_regex", True)

        try:
            self.processor = AutoProcessor.from_pretrained(
                config.model_id,
                trust_remote_code=trust_remote_code,
                **tokenizer_kwargs,
            )
        except Exception:
            self.processor = AutoTokenizer.from_pretrained(
                config.model_id,
                trust_remote_code=trust_remote_code,
                **tokenizer_kwargs,
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

            return outputs[0].outputs[0].text.strip(), None, None
        except Exception as error:
            return None, str(error), None

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
        if not records:
            return []

        try:
            batch_size = self.generate_batch_size or len(records)
            generations: list[GenerationResult] = []

            with tqdm(total=len(records), desc="Generating") as progress:
                if self.use_chat:
                    for chunk in chunks(records, batch_size):
                        requests = [self._build_chat_request(record) for record in chunk]
                        outputs = self.llm.chat(
                            requests,
                            sampling_params=self.sampling_params,
                            use_tqdm=False,
                            chat_template_kwargs=self.chat_template_kwargs,
                        )
                        generations.extend((output.outputs[0].text.strip(), None, None) for output in outputs)
                        progress.update(len(chunk))
                else:
                    for chunk in chunks(records, batch_size):
                        requests = [self._build_generation_request(record) for record in chunk]
                        outputs = self.llm.generate(
                            requests,
                            sampling_params=self.sampling_params,
                            use_tqdm=False,
                        )
                        generations.extend((output.outputs[0].text.strip(), None, None) for output in outputs)
                        progress.update(len(chunk))

            return generations
        except Exception as error:
            return [(None, str(error), None) for _record in records]


def chunks[T](items: list[T], size: int) -> list[list[T]]:
    return [items[index : index + size] for index in range(0, len(items), size)]
