from __future__ import annotations

from typing import Any

from PIL import Image

from povisle.backends.base import BaseBackend, GenerationResult
from povisle.configs import ModelConfig
from povisle.preprocess import preprocess

try:
    import torch
except ImportError:  # pragma: no cover - depends on runtime environment
    torch = None


class HuggingFaceBackend(BaseBackend):
    def __init__(self, config: ModelConfig) -> None:
        if torch is None:
            raise RuntimeError("The hf backend requires torch.")

        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForVision2Seq, AutoProcessor

        model_args = config.model_args
        generation = config.generation
        trust_remote_code = model_args.get("trust_remote_code", True)

        self.processor = AutoProcessor.from_pretrained(config.model_id, trust_remote_code=trust_remote_code)
        self.max_new_tokens = generation.get("max_new_tokens", 64)
        self.temperature = generation.get("temperature", 0.0)
        self.top_p = generation.get("top_p", 1.0)
        self.chat_template_kwargs = generation.get("chat_template_kwargs", {})
        self.preprocessing_config = config.preprocessing

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "device_map": model_args.get("device_map", "auto"),
        }
        dtype = resolve_torch_dtype(model_args.get("dtype", "float16"))
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
            model_kwargs["low_cpu_mem_usage"] = True

        load_error = None
        for model_class in (AutoModelForImageTextToText, AutoModelForVision2Seq, AutoModelForCausalLM):
            try:
                self.model = model_class.from_pretrained(config.model_id, **model_kwargs)
                break
            except Exception as error:  # pragma: no cover - model-family dependent
                load_error = error
        else:
            raise RuntimeError(f"Could not load model {config.model_id}: {load_error}")

    def generate(self, image: Any, prompt: str) -> GenerationResult:
        try:
            prompt = preprocess(prompt, self.preprocessing_config)
            prompt_text = build_processor_prompt(
                self.processor,
                prompt,
                self.chat_template_kwargs,
                include_image=image is not None,
            )
            processor_kwargs = {"text": prompt_text, "return_tensors": "pt"}
            if image is not None:
                processor_kwargs["images"] = ensure_rgb_image(image)
            inputs = self.processor(**processor_kwargs)
            inputs = move_inputs_to_device(inputs, model_device(self.model))
            inputs.pop("token_type_ids", None)

            output = self.model.generate(**inputs, **generation_kwargs(self.max_new_tokens, self.temperature, self.top_p))
            prompt_length = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
            decoded = self.processor.batch_decode(output[:, prompt_length:], skip_special_tokens=True)
            return decoded[0].strip(), None
        except Exception as error:
            return None, str(error)


def resolve_torch_dtype(dtype_name: str | None) -> Any:
    if dtype_name in {None, "", "auto", "none"}:
        return None
    dtype = getattr(torch, str(dtype_name), None)
    if dtype is None:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return dtype


def model_device(model: Any) -> Any:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def move_inputs_to_device(inputs: Any, device: Any) -> Any:
    return inputs.to(device) if hasattr(inputs, "to") else inputs


def generation_kwargs(max_new_tokens: int, temperature: float, top_p: float) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    else:
        kwargs["do_sample"] = False
    return kwargs


def ensure_rgb_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict) and image.get("path"):
        return Image.open(image["path"]).convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(image)!r}")


def build_processor_prompt(
    processor: Any,
    prompt: str,
    chat_template_kwargs: dict[str, Any] | None = None,
    include_image: bool = True,
) -> str:
    chat_template_kwargs = chat_template_kwargs or {}
    multimodal_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    string_messages = [{"role": "user", "content": prompt}]
    template_candidates = (multimodal_messages, string_messages) if include_image else (string_messages,)

    if hasattr(processor, "apply_chat_template"):
        for messages in template_candidates:
            try:
                return processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **chat_template_kwargs,
                )
            except TypeError:
                try:
                    return processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        **chat_template_kwargs,
                    )
                except Exception:
                    continue
            except Exception:
                continue
    return prompt
