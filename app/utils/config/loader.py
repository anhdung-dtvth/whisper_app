"""
Configuration loader for WhisperSign application.

Loads and validates app_config.yaml into a structured dictionary.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Default config values (used when keys are missing from YAML)
_DEFAULTS = {
    "hardware": {
        "device": "leap_motion",
        "target_fps": 60,
        "buffer_duration": 3.0,
    },
    "model": {
        "checkpoint_path": "../Whisper_modification/checkpoints/final_model.pt",
        "vocab_path": "../recorder/data/processed/label_map.json",
        "device": "cuda",
        "window_duration": 3.0,
        "inference_interval": 0.5,
    },
    "preprocessing": {
        "smoothing_window": 5,
        "spatial_normalization": True,
        "scale_normalization": True,
    },
    "ui": {
        "window_width": 1200,
        "window_height": 800,
        "visualization_fps": 30,
        "show_skeleton": True,
        "show_confidence": True,
    },
    "logging": {
        "level": "INFO",
        "file": "logs/app.log",
    },
}


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file, applying defaults for missing keys.

    Args:
        config_path: Path to the YAML config file. If None, tries
                     the default location (whisper_app/config/app_config.yaml).

    Returns:
        Merged configuration dictionary.
    """
    if config_path is None:
        # Default to whisper_app/config/app_config.yaml
        config_path = str(_PROJECT_ROOT / "config" / "app_config.yaml")

    config = {}
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded config from {path}")
    else:
        logger.warning(f"Config file not found at {path}; using defaults.")

    # Deep merge with defaults
    merged = _deep_merge(_DEFAULTS, config)
    _resolve_model_paths(merged, config_dir=path.parent)
    return merged


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    """Recursively merge overrides into defaults."""
    result = defaults.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_model_paths(config: Dict[str, Any], config_dir: Path) -> None:
    """Resolve relative model file paths to stable absolute paths."""
    model_cfg = config.get("model", {})
    for key in ("checkpoint_path", "vocab_path"):
        raw = model_cfg.get(key)
        if not raw:
            continue
        resolved = _resolve_path(str(raw), config_dir=config_dir)
        model_cfg[key] = str(resolved)


def _resolve_path(raw_path: str, config_dir: Path) -> Path:
    """
    Resolve a possibly relative path.

    Priority for relative paths:
      1) Relative to the config file directory (common config behavior)
      2) Relative to the whisper_app project root (legacy behavior)
    """
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    by_config = (config_dir / path).resolve()
    if by_config.exists():
        return by_config

    return (_PROJECT_ROOT / path).resolve()
