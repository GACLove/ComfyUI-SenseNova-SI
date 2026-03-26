from __future__ import annotations

from typing import Any

from .utils import (
    build_generation_kwargs,
    comfy_image_to_pil_images,
    load_qwen_model_class,
    normalize_question,
    resolve_generation_config_path,
    resolve_repo_path,
)

_MODEL_CACHE: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}


class SenseNovaQwenLoader:
    CATEGORY = "SenseNova-SI"
    FUNCTION = "load_model"
    RETURN_TYPES = ("SENSENOVA_QWEN_MODEL",)
    RETURN_NAMES = ("model",)
    SEARCH_ALIASES = ("SenseNova", "Qwen", "Loader")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (
                    "STRING",
                    {
                        "default": "sensenova/SenseNova-SI-1.1-Qwen3-VL-8B",
                        "multiline": False,
                    },
                ),
                "repo_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
                "dtype": (
                    ["auto", "bfloat16", "float16", "float32"],
                    {"default": "auto"},
                ),
                "device_map": (
                    "STRING",
                    {
                        "default": "auto",
                        "multiline": False,
                    },
                ),
                "generation_config_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
                "force_reload": ("BOOLEAN", {"default": False}),
            }
        }

    def load_model(
        self,
        model_path: str,
        repo_path: str,
        dtype: str,
        device_map: str,
        generation_config_path: str,
        force_reload: bool,
    ):
        repo_root = resolve_repo_path(repo_path)
        resolved_generation_config = resolve_generation_config_path(
            repo_root, generation_config_path
        )
        cache_key = (
            str(repo_root),
            model_path,
            dtype,
            device_map,
            resolved_generation_config or "",
        )

        if force_reload or cache_key not in _MODEL_CACHE:
            model_class = load_qwen_model_class(repo_root)
            model = model_class(
                model_path=model_path,
                generation_config=resolved_generation_config,
                device_map=device_map,
                dtype=dtype,
            )
            _MODEL_CACHE[cache_key] = {
                "model": model,
                "repo_root": str(repo_root),
                "model_path": model_path,
            }

        return (_MODEL_CACHE[cache_key],)


class SenseNovaQwenGenerate:
    CATEGORY = "SenseNova-SI"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    SEARCH_ALIASES = ("SenseNova", "Qwen", "Generate")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SENSENOVA_QWEN_MODEL", {"forceInput": True}),
                "question": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "max_new_tokens": ("INT", {"default": 8192, "min": 1, "max": 65536}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.05},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05},
                ),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 16}),
                "do_sample": ("BOOLEAN", {"default": False}),
                "extra_generation_kwargs_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    def generate(
        self,
        model: dict[str, Any],
        question: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        num_beams: int,
        do_sample: bool,
        extra_generation_kwargs_json: str,
        image=None,
    ):
        if not isinstance(model, dict) or "model" not in model:
            raise ValueError(
                "Expected a SENSENOVA_QWEN_MODEL handle from SenseNova Qwen Loader."
            )

        pil_images = comfy_image_to_pil_images(image)
        normalized_question = normalize_question(question, len(pil_images))
        generation_kwargs = build_generation_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
            do_sample=do_sample,
            extra_generation_kwargs_json=extra_generation_kwargs_json,
        )
        response = model["model"].generate(
            normalized_question,
            images=pil_images or None,
            **generation_kwargs,
        )
        return (response,)


NODE_CLASS_MAPPINGS = {
    "SenseNovaQwenLoader": SenseNovaQwenLoader,
    "SenseNovaQwenGenerate": SenseNovaQwenGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SenseNovaQwenLoader": "SenseNova Qwen Loader",
    "SenseNovaQwenGenerate": "SenseNova Qwen Generate",
}
