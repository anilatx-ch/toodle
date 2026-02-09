"""CLI wrapper for Stage 3 traditional ML training."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.run_training import main


if __name__ == "__main__":
    raise SystemExit(main())
