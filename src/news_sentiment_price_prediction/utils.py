from __future__ import annotations

import os
from pathlib import Path
from typing import Union


def ensure_directory(path: Union[str, Path]) -> Path:
    """Create directory if it does not exist and return a Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
