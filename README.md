# ComfyUI-SenseNova-SI

ComfyUI custom node wrapper for [SenseNova-SI](https://github.com/OpenSenseNova/SenseNova-SI) VLM inference.

## Layout

```text
ComfyUI/custom_nodes/ComfyUI-SenseNova-SI/
  __init__.py
  nodes.py
  utils.py
  deps/
    SenseNova-SI/
```

This custom node expects a source checkout of [SenseNova-SI](https://github.com/OpenSenseNova/SenseNova-SI). By default it looks for:

1. `repo_path` from the loader node
2. `SENSENOVA_SI_REPO` environment variable
3. `deps/SenseNova-SI/`
4. The parent directory of this custom node, which makes local development inside the upstream repo work

## Install

1. Clone ComfyUI:

   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   ```

2. Clone this custom node:

   ```bash
   cd custom_nodes
   git clone https://github.com/GACLove/ComfyUI-SenseNova-SI.git
   ```

3. Clone or link [SenseNova-SI](https://github.com/OpenSenseNova/SenseNova-SI) into `deps/`:

   ```bash
   cd ComfyUI-SenseNova-SI
   git clone https://github.com/OpenSenseNova/SenseNova-SI.git deps/SenseNova-SI
   ```

4. Install dependencies:

   ```bash
   uv export --no-hashes > requirements.txt
   pip install -r requirements.txt
   ```

5. Start ComfyUI:

   ```bash
   cd ../../..  # back to ComfyUI root
   python main.py --listen 0.0.0.0 --port 8000 --cuda-device 0
   ```

## Nodes

### SenseNova SI Loader

Loads a SenseNova-SI model once and caches it by:

- repo path
- model path
- dtype
- device map
- generation config path

Use `force_reload` when you need to rebuild the model object.

### SenseNova SI Generate

Runs repeated inference against the loaded model handle.

- `IMAGE` inputs are converted from ComfyUI tensors `[B, H, W, C]` into `PIL.Image`
- if the prompt does not contain `<image>`, the node prepends one token per image
- if the prompt already contains `<image>`, the count must match the image batch size
- `extra_generation_kwargs_json` overrides the exposed generation settings

## Notes

- Supports Qwen and InternVL model types (auto-detected from model path).
- The upstream [SenseNova-SI](https://github.com/OpenSenseNova/SenseNova-SI) repo is used as source, not installed as a Python package.

## License

This project is licensed under the [MIT License](LICENSE).
