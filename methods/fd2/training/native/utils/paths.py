"""Centralized path configuration loaded from methods/fd2/configs/paths.yaml.

Usage:
    from methods.fd2.training.native.utils.paths import PATHS
    raw_data = PATHS["raw_data"]
    deps = PATHS["deps"]
    outputs = PATHS["outputs"]
"""

from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[5]  # repository root
_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"  # methods/fd2/configs
_paths_yaml = _CONFIG_DIR / "paths.yaml"
if not _paths_yaml.exists():
    _paths_yaml = _CONFIG_DIR / "paths_default.yaml"

with open(_paths_yaml) as f:
    _cfg = yaml.safe_load(f)

PATHS = {k: (Path(v) if Path(v).is_absolute() else _PROJECT_ROOT / v)
         for k, v in _cfg.get("dirs", {}).items()}
