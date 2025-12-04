import yaml
from types import SimpleNamespace
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # adjust if depth differs
OPTIMIZERS = {
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
}

def _resolve_paths_refs(obj, paths_dict):
    """
    Recursively resolve strings that contain patterns like "${paths.KEY}",
    including when there is extra text before/after (e.g. "${paths.root}/foo.json").
    """
    if isinstance(obj, dict):
        return {k: _resolve_paths_refs(v, paths_dict) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_paths_refs(v, paths_dict) for v in obj]
    elif isinstance(obj, str):
        # Replace *all* occurrences of ${paths.KEY} in the string
        out = obj
        while "${paths." in out:
            start = out.index("${paths.")
            end = out.index("}", start)
            key = out[start + len("${paths.") : end]
            if key not in paths_dict:
                raise KeyError(f"paths.{key} is not defined in paths.yaml")
            value = str(paths_dict[key])
            out = out[:start] + value + out[end + 1 :]
        return out
    else:
        return obj


def load_config(
    config_path: str,
    paths_path: str = "configs/paths.yaml",
) -> SimpleNamespace:
    """
    Load a YAML config file into a SimpleNamespace, with support for:
      - shared paths.yaml
      - interpolation of ${paths.KEY} in config values
    Paths are resolved relative to the project root.
    """
    config_path = PROJECT_ROOT / config_path
    paths_path = PROJECT_ROOT / paths_path

    with open(paths_path, "r") as f:
        paths = yaml.safe_load(f) or {}

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    cfg_resolved = _resolve_paths_refs(cfg, paths)
    cfg_resolved["paths"] = SimpleNamespace(**paths)

    return SimpleNamespace(**cfg_resolved)
