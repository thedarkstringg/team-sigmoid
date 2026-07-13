"""
Entry-point wrapper so that `python src/experiments/run_all.py` works
from the repository root (as required by the project spec).

Adds the repo root to sys.path, then delegates to experiments.run_all.main().
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../src/experiments/
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))  # repo root
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.run_all import main  # noqa: E402

if __name__ == "__main__":
    main()
