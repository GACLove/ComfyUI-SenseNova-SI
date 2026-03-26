from __future__ import annotations

from typing import Any

import torch

from .utils import (
    build_generation_kwargs,
    comfy_image_to_pil_images,
    load_model_class,
    normalize_question,
    pil_images_to_temp_paths,
    resolve_generation_config_path,
    resolve_model_type,
    resolve_repo_path,
)

_MODEL_CACHE: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}


class SenseNovaSILoader:
    CATEGORY = "SenseNova-SI"
    FUNCTION = "load_model"
    RETURN_TYPES = ("SENSENOVA_SI_MODEL",)
    RETURN_NAMES = ("model",)
    SEARCH_ALIASES = ("SenseNova", "SI", "Loader")

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
                "model_type": (
                    ["auto", "qwen", "internvl"],
                    {"default": "auto"},
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
        model_type: str,
        repo_path: str,
        dtype: str,
        device_map: str,
        generation_config_path: str,
        force_reload: bool,
    ):
        resolved_type = resolve_model_type(model_type, model_path)
        repo_root = resolve_repo_path(repo_path)
        resolved_generation_config = resolve_generation_config_path(repo_root, generation_config_path)
        cache_key = (
            str(repo_root),
            model_path,
            resolved_type,
            dtype,
            device_map,
            resolved_generation_config or "",
        )

        if force_reload or cache_key not in _MODEL_CACHE:
            model_class = load_model_class(repo_root, resolved_type)
            if resolved_type == "qwen":
                kwargs = {"model_path": model_path, "device_map": device_map, "dtype": dtype}
                if resolved_generation_config:
                    kwargs["generation_config"] = resolved_generation_config
                model = model_class(**kwargs)
            else:
                kwargs = {"model_path": model_path}
                if resolved_generation_config:
                    kwargs["generation_config"] = resolved_generation_config
                model = model_class(**kwargs)
            _MODEL_CACHE[cache_key] = {
                "model": model,
                "model_type": resolved_type,
                "repo_root": str(repo_root),
                "model_path": model_path,
            }

        return (_MODEL_CACHE[cache_key],)


class SenseNovaSIGenerate:
    CATEGORY = "SenseNova-SI"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    SEARCH_ALIASES = ("SenseNova", "SI", "Generate")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SENSENOVA_SI_MODEL", {"forceInput": True}),
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
            raise ValueError("Expected a SENSENOVA_SI_MODEL handle from SenseNova SI Loader.")

        model_type = model.get("model_type", "qwen")
        pil_images = comfy_image_to_pil_images(image)
        generation_kwargs = build_generation_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
            do_sample=do_sample,
            extra_generation_kwargs_json=extra_generation_kwargs_json,
        )

        if model_type == "qwen":
            normalized_question = normalize_question(question, len(pil_images))
            response = model["model"].generate(
                normalized_question,
                images=pil_images or None,
                **generation_kwargs,
            )
        else:
            # internvl expects file paths
            if pil_images:
                image_paths = pil_images_to_temp_paths(pil_images)
            else:
                image_paths = None
            response = model["model"].generate(
                question,
                images=image_paths,
                **generation_kwargs,
            )

        return (response,)


_MAX_IMAGE_INPUTS = 10


class SenseNovaSIImageList:
    """Aggregates up to 10 individual IMAGE inputs into a single batched IMAGE."""

    CATEGORY = "SenseNova-SI"
    FUNCTION = "aggregate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    SEARCH_ALIASES = ("SenseNova", "Image", "List", "Aggregate")

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, _MAX_IMAGE_INPUTS + 1):
            optional[f"image_{i}"] = ("IMAGE",)
        return {"required": {}, "optional": optional}

    def aggregate(self, **kwargs):
        images = []
        for i in range(1, _MAX_IMAGE_INPUTS + 1):
            img = kwargs.get(f"image_{i}")
            if img is not None:
                images.append(img)

        if not images:
            raise ValueError("At least one image input must be connected.")

        return (torch.cat(images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "SenseNovaSILoader": SenseNovaSILoader,
    "SenseNovaSIGenerate": SenseNovaSIGenerate,
    "SenseNovaSIImageList": SenseNovaSIImageList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SenseNovaSILoader": "SenseNova SI Loader",
    "SenseNovaSIGenerate": "SenseNova SI Generate",
    "SenseNovaSIImageList": "SenseNova Image List",
}
