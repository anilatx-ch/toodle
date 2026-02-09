"""Orchestrate Stage 3 traditional ML training runs."""

from __future__ import annotations

import argparse
from typing import Any, Callable, Sequence

from src.training.train_catboost import train_catboost_model
from src.training.train_xgboost import train_xgboost_model

Runner = Callable[..., dict[str, Any]]


def _selected_models(choice: str) -> list[str]:
    if choice == "all":
        return ["catboost", "xgboost"]
    return [choice]


def _runner_map() -> dict[str, Runner]:
    return {
        "catboost": train_catboost_model,
        "xgboost": train_xgboost_model,
    }


def run_training(
    *,
    model_choice: str,
    enable_optuna: bool,
    continue_on_error: bool,
) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str]]]:
    """Run one or more training jobs and collect success/error payloads."""
    results: dict[str, dict[str, Any]] = {}
    errors: list[tuple[str, str]] = []

    runners = _runner_map()
    for model_name in _selected_models(model_choice):
        runner = runners[model_name]
        try:
            results[model_name] = runner(enable_optuna=enable_optuna)
            print(
                f"{model_name} complete: "
                f"f1_weighted={results[model_name]['metrics']['f1_weighted']:.4f}"
            )
        except Exception as exc:
            message = str(exc)
            errors.append((model_name, message))
            print(f"[ERROR] {model_name} failed: {message}")
            if not continue_on_error:
                break

    return results, errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 3 traditional ML training")
    parser.add_argument("--model", choices=["catboost", "xgboost", "all"], default="all")
    parser.add_argument("--optuna", action="store_true", help="Enable Optuna tuning")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)

    results, errors = run_training(
        model_choice=args.model,
        enable_optuna=args.optuna,
        continue_on_error=args.continue_on_error,
    )

    if not results and errors:
        return 1
    if errors:
        return 1

    print(f"Training completed for: {', '.join(results.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
