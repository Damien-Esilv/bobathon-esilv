"""Utilities for loading Skore Hub credentials from .skore."""
from __future__ import annotations

import json
from pathlib import Path


def load_skore_credentials() -> dict:
    """Return the parsed contents of the repo-root .skore file."""
    skore_path = Path(__file__).resolve().parents[2] / ".skore"
    return json.loads(skore_path.read_text())
