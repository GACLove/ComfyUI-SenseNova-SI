from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

IMAGE_TOKEN = "<image>"
DEFAULT_REPO_ENV_VAR = "SENSENOVA_SI_REPO"

_LOADED_REPO_ROOT: Path | None = None


def _is_repo_root(path: Path) -> bool:
    return (path / "sensenova_si" / "__init__.py").exists()


def resolve_repo_path(repo_path: str = "") -> Path:
    package_root = Path(__file__).resolve().parent
    candidate_strings = [
        repo_path.strip(),
        os.getenv(DEFAULT_REPO_ENV_VAR, "").strip(),
    ]
    candidates = [Path(candidate).expanduser() for candidate in candidate_strings if candidate]
    candidates.extend(
        [
            package_root / "deps" / "SenseNova-SI",
            package_root.parent,
        ]
    )

    tried_paths = []
    for candidate in candidates:
        resolved = candidate.resolve()
        tried_paths.append(str(resolved))
        if _is_repo_root(resolved):
            return resolved

    raise FileNotFoundError(
        "Unable to locate a SenseNova-SI source checkout. "
        f"Set `{DEFAULT_REPO_ENV_VAR}`, fill `repo_path`, or clone the repo into "
        f"`{package_root / 'deps' / 'SenseNova-SI'}`. Tried: {', '.join(tried_paths)}"
    )


def resolve_generation_config_path(repo_root: Path, generation_config_path: str = "") -> str | None:
    value = generation_config_path.strip()
    if not value:
        return None

    path = Path(value).expanduser()
    if not path.is_absolute():
        return str((repo_root / path).resolve())
    return str(path.resolve())


def _ensure_repo_on_path(repo_root: Path):
    global _LOADED_REPO_ROOT

    repo_root = repo_root.resolve()
    if _LOADED_REPO_ROOT != repo_root:
        for module_name in list(sys.modules):
            if module_name == "sensenova_si" or module_name.startswith("sensenova_si."):
                sys.modules.pop(module_name)
        sys.path = [entry for entry in sys.path if Path(entry).resolve() != repo_root]
        sys.path.insert(0, str(repo_root))
        importlib.invalidate_caches()
        _LOADED_REPO_ROOT = repo_root


_MODEL_TYPE_MAP = {
    "qwen": ("sensenova_si.qwen", "SenseNovaSIQwenModel"),
    "internvl": ("sensenova_si.internvl", "SenseNovaSIInternVLModel"),
}


def resolve_model_type(model_type: str, model_path: str) -> str:
    if model_type != "auto":
        return model_type
    lower = model_path.lower()
    if "qwen" in lower:
        return "qwen"
    if "internvl" in lower:
        return "internvl"
    return "qwen"


def load_model_class(repo_root: Path, model_type: str):
    _ensure_repo_on_path(repo_root)
    module_name, class_name = _MODEL_TYPE_MAP[model_type]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_qwen_model_class(repo_root: Path):
    return load_model_class(repo_root, "qwen")


def comfy_image_to_pil_images(image: torch.Tensor | None) -> list[Image.Image]:
    if image is None:
        return []
    if image.ndim != 4:
        raise ValueError(f"Expected ComfyUI IMAGE tensor with shape [B, H, W, C], got {tuple(image.shape)}")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected ComfyUI IMAGE tensor with 3 channels, got last dim {image.shape[-1]}")

    pil_images = []
    image = image.detach().cpu().clamp(0, 1)
    for frame in image:
        array = np.clip(frame.numpy() * 255.0, 0, 255).astype(np.uint8)
        pil_images.append(Image.fromarray(array, mode="RGB"))
    return pil_images


def normalize_question(question: str, image_count: int) -> str:
    image_token_count = question.count(IMAGE_TOKEN)
    if image_count == 0:
        if image_token_count:
            raise ValueError(
                f"Prompt contains {image_token_count} `{IMAGE_TOKEN}` token(s), but no image input is connected."
            )
        return question

    if image_token_count == 0:
        prefix = "\n".join([IMAGE_TOKEN] * image_count)
        return f"{prefix}\n{question}".strip()

    if image_token_count != image_count:
        raise ValueError(
            f"Prompt contains {image_token_count} `{IMAGE_TOKEN}` token(s), but received {image_count} image(s)."
        )
    return question


def pil_images_to_temp_paths(pil_images: list[Image.Image]) -> list[str]:
    """Save PIL images to temporary files and return their paths.

    Used for InternVL which expects file paths instead of PIL images.
    """
    import tempfile

    paths = []
    for i, img in enumerate(pil_images):
        fd, path = tempfile.mkstemp(suffix=f"_{i}.png")
        os.close(fd)
        img.save(path)
        paths.append(path)
    return paths


def build_generation_kwargs(
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    num_beams: int,
    do_sample: bool,
    extra_generation_kwargs_json: str,
) -> dict[str, Any]:
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "num_beams": num_beams,
        "do_sample": do_sample,
    }

    extra_value = extra_generation_kwargs_json.strip()
    if not extra_value:
        return generation_kwargs

    try:
        extra_kwargs = json.loads(extra_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid extra_generation_kwargs_json: {exc}") from exc
    if not isinstance(extra_kwargs, dict):
        raise ValueError("extra_generation_kwargs_json must decode to a JSON object.")

    generation_kwargs.update(extra_kwargs)
    return generation_kwargs
