"""Active experiment settings.

Other modules import this file. load() copies one configs/<name>.py onto it.
Importing config loads the A1 experiment, which is the default run.
"""

import importlib.util
from pathlib import Path

_CONFIGS = Path(__file__).resolve().parent / "configs"


def load(name):
    """Load configs/<name>.py and copy its uppercase settings onto this module."""
    path = _CONFIGS / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"No experiment config at {path}")
    spec = importlib.util.spec_from_file_location(f"configs.{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key, value in vars(module).items():
        if key.isupper():
            globals()[key] = value
    return module


load("a1")
