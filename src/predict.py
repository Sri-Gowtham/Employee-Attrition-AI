"""
predict.py
----------
Loads a trained Employee Attrition pipeline and runs predictions.

Supports three modes:
  1. Single employee  — pass values as CLI flags
  2. Batch CSV        — pass a CSV file with --input-csv
  3. Interactive REPL — run without arguments for a prompt-driven session

Usage
-----
    # Single prediction (all fields required):
    python src/predict.py \\
        --satisfaction-level 0.38 \\
        --last-evaluation 0.53 \\
        --number-project 2 \\
        --average-montly-hours 157 \\
        --time-spend-company 3 \\
        --work-accident 0 \\
        --promotion-last-5years 0 \\
        --department sales \\
        --salary low

    # Batch prediction from CSV:
    python src/predict.py --input-csv new_employees.csv --output-csv predictions.csv

    # Interactive REPL:
    python src/predict.py

Python 3.11+ required.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure src/ is importable regardless of working directory
# (must come before any third-party or local imports)
sys.path.insert(0, str(Path(__file__).parent))

# Force UTF-8 on Windows consoles (fixes cp1252 UnicodeEncodeError with emoji)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from data_preprocessing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    EXPECTED_DEPARTMENTS,
    EXPECTED_SALARY_LEVELS,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH: Path = (
    Path(__file__).parent.parent / "models" / "attrition_rf_pipeline.joblib"
)

LABEL_MAP: dict[int, str] = {0: "[STAY]  Will Stay", 1: "[RISK]  At Risk of Leaving"}

# Human-readable column descriptions for the interactive REPL
FIELD_PROMPTS: dict[str, dict[str, Any]] = {
    "satisfaction_level": {
        "prompt": "Satisfaction level [0.0 – 1.0]",
        "type": float,
        "valid": lambda v: 0.0 <= v <= 1.0,
        "hint": "e.g. 0.72",
    },
    "last_evaluation": {
        "prompt": "Last evaluation score [0.0 – 1.0]",
        "type": float,
        "valid": lambda v: 0.0 <= v <= 1.0,
        "hint": "e.g. 0.87",
    },
    "number_project": {
        "prompt": "Number of projects assigned",
        "type": int,
        "valid": lambda v: 1 <= v <= 20,
        "hint": "e.g. 5",
    },
    "average_montly_hours": {
        "prompt": "Average monthly hours worked",
        "type": int,
        "valid": lambda v: 50 <= v <= 400,
        "hint": "e.g. 220",
    },
    "time_spend_company": {
        "prompt": "Years spent at the company",
        "type": int,
        "valid": lambda v: 1 <= v <= 40,
        "hint": "e.g. 4",
    },
    "Work_accident": {
        "prompt": "Had a workplace accident? [0 = No, 1 = Yes]",
        "type": int,
        "valid": lambda v: v in (0, 1),
        "hint": "0 or 1",
    },
    "promotion_last_5years": {
        "prompt": "Promoted in the last 5 years? [0 = No, 1 = Yes]",
        "type": int,
        "valid": lambda v: v in (0, 1),
        "hint": "0 or 1",
    },
    "department": {
        "prompt": f"Department {list(EXPECTED_DEPARTMENTS)}",
        "type": str,
        "valid": lambda v: v.lower() in [d.lower() for d in EXPECTED_DEPARTMENTS],
        "hint": "e.g. sales",
        "transform": lambda v: v.strip(),
    },
    "salary": {
        "prompt": f"Salary level {list(EXPECTED_SALARY_LEVELS)}",
        "type": str,
        "valid": lambda v: v.lower() in EXPECTED_SALARY_LEVELS,
        "hint": "low / medium / high",
        "transform": lambda v: v.strip().lower(),
    },
}


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------


def load_pipeline(model_path: str | Path) -> Pipeline:
    """Load a serialised sklearn Pipeline from disk.

    Parameters
    ----------
    model_path:
        Path to the .joblib file produced by train_model.py.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted pipeline ready for inference.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path.resolve()}\n"
            "Run 'python src/train_model.py' first to train and save the model."
        )
    pipeline: Pipeline = joblib.load(path)
    logger.info("Model loaded from: %s", path.resolve())
    return pipeline


# ---------------------------------------------------------------------------
# Core Prediction Logic
# ---------------------------------------------------------------------------


def predict_single(
    pipeline: Pipeline,
    employee: dict[str, Any],
) -> dict[str, Any]:
    """Predict attrition risk for a single employee record.

    Parameters
    ----------
    pipeline:
        Fitted sklearn Pipeline.
    employee:
        Dictionary with keys matching ALL_FEATURES.

    Returns
    -------
    dict with keys:
        - prediction (int): 0 = stays, 1 = at risk
        - label (str): human-readable outcome
        - probability_stay (float): P(stays)
        - probability_quit (float): P(at risk)
    """
    _validate_employee_dict(employee)
    X = pd.DataFrame([employee])[ALL_FEATURES]

    prediction: int = int(pipeline.predict(X)[0])
    probabilities: np.ndarray = pipeline.predict_proba(X)[0]

    return {
        "prediction": prediction,
        "label": LABEL_MAP[prediction],
        "probability_stay": float(probabilities[0]),
        "probability_quit": float(probabilities[1]),
    }


def predict_batch(
    pipeline: Pipeline,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Predict attrition risk for a DataFrame of employees.

    Parameters
    ----------
    pipeline:
        Fitted sklearn Pipeline.
    df:
        DataFrame containing at least ALL_FEATURES columns.
        May also contain the target column 'quit' (will be preserved).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with appended columns:
        'predicted_quit', 'probability_stay', 'probability_quit', 'risk_label'.
    """
    missing = set(ALL_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    X = df[ALL_FEATURES].copy()

    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)

    result = df.copy()
    result["predicted_quit"] = predictions
    result["probability_stay"] = probabilities[:, 0]
    result["probability_quit"] = probabilities[:, 1]
    result["risk_label"] = [LABEL_MAP[p] for p in predictions]

    at_risk = int((predictions == 1).sum())
    logger.info(
        "Batch prediction complete — %d / %d employees at risk of leaving (%.1f%%)",
        at_risk, len(df), 100.0 * at_risk / len(df),
    )
    return result


def _validate_employee_dict(employee: dict[str, Any]) -> None:
    """Raise ValueError if any required field is missing."""
    missing = set(ALL_FEATURES) - set(employee.keys())
    if missing:
        raise ValueError(
            f"Employee record is missing required fields: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------


def interactive_predict(pipeline: Pipeline) -> None:
    """Run an interactive command-line prompt for single predictions."""
    print("\n" + "=" * 60)
    print("  Employee Attrition Predictor — Interactive Mode")
    print("=" * 60)
    print("  Enter 'q' at any prompt to quit.\n")

    while True:
        employee: dict[str, Any] = {}

        for field, meta in FIELD_PROMPTS.items():
            while True:
                raw = input(f"  {meta['prompt']}  ({meta['hint']}): ").strip()
                if raw.lower() == "q":
                    print("\n  Goodbye!\n")
                    return

                transform = meta.get("transform", lambda v: v)
                try:
                    value = meta["type"](transform(raw))
                    if not meta["valid"](value):
                        raise ValueError(f"Value '{value}' is out of range.")
                    employee[field] = value
                    break
                except (ValueError, TypeError) as exc:
                    print(f"    ⚠ Invalid input: {exc}  — please try again.")

        result = predict_single(pipeline, employee)
        print("\n" + "─" * 60)
        print(f"  PREDICTION     : {result['label']}")
        print(f"  Prob(Stays)    : {result['probability_stay']:.2%}")
        print(f"  Prob(Leaves)   : {result['probability_quit']:.2%}")
        print("─" * 60 + "\n")

        cont = input("  Predict another employee? [y/N]: ").strip().lower()
        if cont != "y":
            print("\n  Goodbye!\n")
            break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Employee Attrition predictor using a trained pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained .joblib pipeline.",
    )

    # ── Batch mode ──
    batch_group = parser.add_argument_group("Batch prediction")
    batch_group.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="CSV file with employee records for batch prediction.",
    )
    batch_group.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Where to save batch prediction results (default: print to stdout).",
    )

    # ── Single prediction mode ──
    single_group = parser.add_argument_group("Single employee prediction")
    single_group.add_argument("--satisfaction-level", type=float, dest="satisfaction_level")
    single_group.add_argument("--last-evaluation", type=float, dest="last_evaluation")
    single_group.add_argument("--number-project", type=int, dest="number_project")
    single_group.add_argument("--average-montly-hours", type=int, dest="average_montly_hours")
    single_group.add_argument("--time-spend-company", type=int, dest="time_spend_company")
    single_group.add_argument("--work-accident", type=int, dest="Work_accident", choices=[0, 1])
    single_group.add_argument("--promotion-last-5years", type=int, dest="promotion_last_5years", choices=[0, 1])
    single_group.add_argument("--department", type=str, dest="department", choices=EXPECTED_DEPARTMENTS)
    single_group.add_argument("--salary", type=str, dest="salary", choices=EXPECTED_SALARY_LEVELS)

    return parser.parse_args()


def _all_single_args_provided(args: argparse.Namespace) -> bool:
    """Return True if all single-prediction CLI arguments were supplied."""
    single_fields = [
        "satisfaction_level", "last_evaluation", "number_project",
        "average_montly_hours", "time_spend_company", "Work_accident",
        "promotion_last_5years", "department", "salary",
    ]
    return all(getattr(args, f, None) is not None for f in single_fields)


def main() -> None:
    args = parse_args()

    # ── Load model ──
    pipeline = load_pipeline(args.model)

    # ── Batch mode ──
    if args.input_csv is not None:
        if not args.input_csv.exists():
            logger.error("Input CSV not found: %s", args.input_csv)
            sys.exit(1)

        logger.info("Batch prediction mode — reading: %s", args.input_csv)
        df = pd.read_csv(args.input_csv)
        result_df = predict_batch(pipeline, df)

        if args.output_csv:
            args.output_csv.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(args.output_csv, index=False)
            logger.info("Results saved → %s", args.output_csv.resolve())
        else:
            print(result_df[
                ["predicted_quit", "probability_stay", "probability_quit", "risk_label"]
            ].to_string(index=False))
        return

    # ── Single prediction mode ──
    if _all_single_args_provided(args):
        employee = {
            "satisfaction_level": args.satisfaction_level,
            "last_evaluation": args.last_evaluation,
            "number_project": args.number_project,
            "average_montly_hours": args.average_montly_hours,
            "time_spend_company": args.time_spend_company,
            "Work_accident": args.Work_accident,
            "promotion_last_5years": args.promotion_last_5years,
            "department": args.department,
            "salary": args.salary,
        }
        result = predict_single(pipeline, employee)
        print(f"\nPrediction : {result['label']}")
        print(f"Prob(Stay) : {result['probability_stay']:.2%}")
        print(f"Prob(Quit) : {result['probability_quit']:.2%}\n")
        return

    # ── Interactive REPL (default when no args given) ──
    interactive_predict(pipeline)


if __name__ == "__main__":
    main()
