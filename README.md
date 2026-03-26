# ComfyUI-SenseNova-SI

ComfyUI custom node wrapper for `SenseNova-SI` Qwen-based VLM inference.

## Layout

```text
ComfyUI/custom_nodes/ComfyUI-SenseNova-SI/
  __init__.py
  nodes.py
  utils.py
  deps/
    SenseNova-SI/
```

This custom node expects a source checkout of `SenseNova-SI`. By default it looks for:

1. `repo_path` from the loader node
2. `SENSENOVA_SI_REPO` environment variable
3. `deps/SenseNova-SI/`
4. The parent directory of this custom node, which makes local development inside the upstream repo work

## Install

1. Place this directory in `ComfyUI/custom_nodes/`.
2. Clone the upstream repo into `ComfyUI/custom_nodes/ComfyUI-SenseNova-SI/deps/SenseNova-SI`.
3. Install the runtime dependencies from `requirements.txt` into the ComfyUI Python environment.
4. Restart ComfyUI.

## Nodes

### SenseNova Qwen Loader

Loads a `SenseNovaSIQwenModel` once and caches it by:

- repo path
- model path
- dtype
- device map
- generation config path

Use `force_reload` when you need to rebuild the model object.

### SenseNova Qwen Generate

Runs repeated inference against the loaded model handle.

- `IMAGE` inputs are converted from ComfyUI tensors `[B, H, W, C]` into `PIL.Image`
- if the prompt does not contain `<image>`, the node prepends one token per image
- if the prompt already contains `<image>`, the count must match the image batch size
- `extra_generation_kwargs_json` overrides the exposed generation settings

## Notes

- This wrapper only targets the Qwen path right now.
- The upstream `SenseNova-SI` repo is used as source, not installed as a Python package.

## License

This project is licensed under the [MIT License](LICENSE).
