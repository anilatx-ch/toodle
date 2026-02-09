"""Build and persist anomaly detection baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.anomaly.baselines import build_and_save_baseline  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build anomaly detection baseline")
    parser.add_argument("--source", choices=["training"], default="training")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config.ensure_directories()

    output_path = Path(args.output) if args.output else None
    baseline = build_and_save_baseline(source=args.source, output_path=output_path)
    path = output_path or config.ANOMALY_BASELINE_PATH

    print(f"Saved baseline: {path}")
    print(f"Total predictions: {baseline.total_predictions}")
    print(f"Categories: {len(baseline.category_baselines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

