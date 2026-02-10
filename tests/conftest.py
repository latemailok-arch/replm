"""Shared test fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the src directory is on the path (workaround for spaces in iCloud path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
