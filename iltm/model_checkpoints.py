import os
import sys
from pathlib import Path
from typing import Dict, Any

from huggingface_hub import hf_hub_download

from .hyperparameter_configs import _CHECKPOINT_MIXES


CKPT_DIR_ENV = "ILTM_CKPT_DIR"
HF_REPO_ID = "dbonet/iLTM"
AVAILABLE_CHECKPOINTS = [
    "xgbrconcat", "cbrconcat", "r128bn", "rnobn", "xgb", "catb", "rtr", "rtrcb",
]


def resolve_checkpoint_mix(
    checkpoint_mix: str | tuple[str, ...] | list[str] | None,
) -> tuple[str, ...] | list[str] | None:
    """Expand a named mix while accepting existing explicit checkpoint lists."""
    if checkpoint_mix is None:
        return None
    if isinstance(checkpoint_mix, str):
        if checkpoint_mix != "default":
            raise ValueError(f"Unknown checkpoint_mix preset: {checkpoint_mix!r}.")
        # The eight-member mix is the prefix of this recipe.
        return tuple(_CHECKPOINT_MIXES[12])
    if not isinstance(checkpoint_mix, (list, tuple)):
        raise TypeError("checkpoint_mix must be a preset name, list, or tuple of checkpoints.")
    if not checkpoint_mix:
        raise ValueError("checkpoint_mix must contain at least one checkpoint.")
    return checkpoint_mix


def restrict_checkpoint_choices(
    space: dict[str, dict[str, Any]], available_checkpoints: list[str] | None,
) -> dict[str, dict[str, Any]]:
    """Limit a space to supplied checkpoints, preserving explicit local paths."""
    if available_checkpoints is None:
        return space
    checkpoints = list(available_checkpoints)
    if not checkpoints:
        raise ValueError("available_checkpoints must contain at least one checkpoint.")
    space["checkpoint"] = {"type": "categorical", "choices": checkpoints}

    by_family = {}
    for checkpoint in checkpoints:
        name = str(checkpoint).removesuffix(".pth")
        for family in AVAILABLE_CHECKPOINTS:
            if name.endswith(family):
                by_family.setdefault(family, checkpoint)
                break

    recipe = resolve_checkpoint_mix("default")
    if set(recipe) <= by_family.keys():
        # Custom locations need an explicit recipe so every member uses the
        # supplied file, even when the files live in different directories.
        mix = tuple(by_family[name] for name in recipe)
        choice = "default" if mix == recipe else mix
        space["checkpoint_mix"] = {**space["checkpoint_mix"], "choices": [None, choice]}
    else:
        space["checkpoint_mix"] = {"type": "constant", "value": None}
    return space


def _get_platform_cache_dir() -> Path:
    if sys.platform == "win32":
        appdata = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "iltm"
        return Path.home() / ".iltm"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "iltm"
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache:
            return Path(xdg_cache) / "iltm"
        return Path.home() / ".cache" / "iltm"


def _get_default_ckpt_dir() -> str:
    if CKPT_DIR_ENV in os.environ:
        ckpt_dir = os.environ[CKPT_DIR_ENV]
    else:
        ckpt_dir = str(_get_platform_cache_dir())
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir


def _ensure_checkpoint(repo_id: str, filename: str, ckpt_dir: str) -> str:
    local_path = os.path.join(ckpt_dir, filename)
    if os.path.isfile(local_path):
        return local_path
    
    try:
        from huggingface_hub.utils import HfHubHTTPError
        
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=ckpt_dir,
        )
        return path
    except HfHubHTTPError as e:
        if e.response.status_code in (401, 403):
            error_msg = (
                f"\nFailed to download checkpoint '{filename}' from Hugging Face.\n"
                f"Repository: {repo_id}\n\n"
                f"This appears to be an authentication issue. Try the following:\n\n"
                f"  1. Authenticate with Hugging Face using one of these commands:\n"
                f"     - huggingface-cli login\n"
                f"     - hf auth login\n\n"
                f"  2. If you don't have an account, create one at:\n"
                f"     https://huggingface.co/join\n\n"
            )
            raise RuntimeError(error_msg) from e
        raise
    except ImportError:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=ckpt_dir,
        )
        return path


_CHECKPOINT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "r128bn": {
        "preprocessing": "realmlp_td_s_v0",
        "bottleneck_size": 128,
        "tree_embedding": False,
        "concat_tree_with_orig_features": False,
    },
    "rnobn": {
        "preprocessing": "realmlp_td_s_v0",
        "tree_embedding": False,
        "concat_tree_with_orig_features": False,
    },
    "catb": {
        "preprocessing": "none",
        "tree_embedding": True,
        "tree_model": "CatBoost",
        "concat_tree_with_orig_features": False,
    },
    "xgb": {
        "preprocessing": "none",
        "tree_embedding": True,
        "tree_model": "XGBoost_hist",
        "concat_tree_with_orig_features": False,
    },
    "rtr": {
        "preprocessing": "realmlp_td_s_v0",
        "do_retrieval": True,
        "tree_embedding": False,
        "concat_tree_with_orig_features": False,
    },
    "rtrcb": {
        "preprocessing": "none",
        "tree_embedding": True,
        "tree_model": "CatBoost",
        "do_retrieval": True,
        "concat_tree_with_orig_features": False,
    },
    "cbrconcat": {
        "preprocessing": "realmlp_td_s_v0",
        "tree_embedding": True,
        "tree_model": "CatBoost",
        "concat_tree_with_orig_features": True,
    },
    "xgbrconcat": {
        "preprocessing": "realmlp_td_s_v0",
        "tree_embedding": True,
        "tree_model": "XGBoost_hist",
        "concat_tree_with_orig_features": True,
    },
}


def get_model_checkpoint_config(model_name_suffix: str) -> Dict[str, Any]:
    for model_name, config in _CHECKPOINT_CONFIGS.items():
        if model_name_suffix.endswith(model_name):
            local_path = _ensure_checkpoint(
                HF_REPO_ID, f"{model_name}.pth", _get_default_ckpt_dir(),
            )
            return {**config, "checkpoint": local_path}

    raise ValueError(
        f"Unknown model name suffix: {model_name_suffix}. "
        f"Available models: {list(_CHECKPOINT_CONFIGS)}"
    )


def _local_checkpoint_config(checkpoint: str) -> Dict[str, Any]:
    # A defined checkpoint keeps its identity when moved to another directory.
    # Preserve the supplied path object for sklearn's constructor/clone contract.
    config = _CHECKPOINT_CONFIGS.get(Path(checkpoint).stem, {})
    return {**config, "checkpoint": checkpoint}


def resolve_model_checkpoint(checkpoint: str) -> Dict[str, Any]:
    if os.path.isfile(checkpoint):
        return _local_checkpoint_config(checkpoint)

    if os.path.isdir(checkpoint):
        pth_files = list(Path(checkpoint).glob("*.pth"))
        if pth_files:
            return _local_checkpoint_config(str(pth_files[0]))

    return get_model_checkpoint_config(checkpoint)
