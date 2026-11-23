"""Active experiment settings.

Other modules import this file. load() copies one configs/<name>.py onto it.
Importing config loads the A1 experiment, which is the default run.
"""

import importlib.util
from pathlib import Path

_config_dir = Path(__file__).resolve().parent / "configs"


def load(name):
    """Load configs/<name>.py and copy its uppercase settings onto this module."""
    path = _config_dir / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"No experiment config at {path}")
    spec = importlib.util.spec_from_file_location(f"configs.{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    incoming = {key: value for key, value in vars(module).items() if key.isupper()}
    # Private names such as _config_dir stay. A later config must not keep
    # uppercase settings that the new file does not define.
    stale = [
        key
        for key in list(globals())
        if key.isupper() and not key.startswith("_") and key not in incoming
    ]
    for key in stale:
        del globals()[key]
    globals().update(incoming)
    return module


load("a1")
