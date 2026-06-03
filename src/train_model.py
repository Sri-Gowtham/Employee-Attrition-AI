"""
train_model.py
--------------
Trains a Random Forest classifier for Employee Attrition Prediction
using a full sklearn Pipeline (preprocessing + model).

Usage
-----
    # From the project root:
    python src/train_model.py

    # With custom data path / output dir:
    python src/train_model.py \\
        --data data/employee_data.csv \\
        --output models/ \\
        --test-size 0.20 \\
        --n-estimators 200 \\
        --max-depth 15

What this script does
---------------------
1. Loads and validates employee_data.csv
2. Splits into train / test sets (stratified)
3. Builds a Pipeline:  ColumnTransformer → RandomForestClassifier
4. Runs 5-fold cross-validation on the training set
5. Fits the final model on all training data
6. Evaluates on the held-out test set (accuracy, F1, AUC, classification report)
7. Saves the fitted pipeline to disk with joblib
8. Logs feature importances ranked by contribution

Python 3.11+ required.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

from data_preprocessing import (
    build_preprocessor,
    load_and_prepare,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
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
DEFAULT_DATA_PATH: Path = Path(__file__).parent.parent / "data" / "employee_data.csv"
DEFAULT_OUTPUT_DIR: Path = Path(__file__).parent.parent / "models"
DEFAULT_MODEL_FILENAME: str = "attrition_rf_pipeline.joblib"
DEFAULT_METRICS_FILENAME: str = "evaluation_metrics.json"

# ---------------------------------------------------------------------------
# Model Builder
# ---------------------------------------------------------------------------


def build_pipeline(
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: str = "balanced",
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """Construct a full sklearn Pipeline: preprocessor → Random Forest.

    Parameters
    ----------
    n_estimators:
        Number of trees in the forest.
    max_depth:
        Maximum depth of each tree (None = grow until pure leaves).
    min_samples_split:
        Minimum samples required to split an internal node.
    min_samples_leaf:
        Minimum samples required at a leaf node.
    class_weight:
        'balanced' automatically handles class imbalance by adjusting weights
        inversely proportional to class frequencies.
    random_state:
        Seed for reproducibility.
    n_jobs:
        Number of CPU jobs (-1 = use all available cores).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline ready for .fit() / .predict() calls.
    """
    preprocessor = build_preprocessor()

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        criterion="gini",
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ],
        verbose=False,
    )

    logger.info(
        "Pipeline built — RandomForest(n_estimators=%d, max_depth=%s, "
        "class_weight='%s')",
        n_estimators, max_depth, class_weight,
    )
    return pipeline


# ---------------------------------------------------------------------------
# Cross-Validation
# ---------------------------------------------------------------------------


def cross_validate_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """Run stratified k-fold cross-validation and return mean CV metrics.

    Parameters
    ----------
    pipeline:
        Unfitted pipeline.
    X_train:
        Training feature matrix.
    y_train:
        Training target series.
    n_splits:
        Number of CV folds.
    random_state:
        Seed for StratifiedKFold.

    Returns
    -------
    dict
        Mean and std of accuracy, F1 (weighted), and ROC-AUC across folds.
    """
    logger.info("Running %d-fold stratified cross-validation …", n_splits)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "f1_weighted": "f1_weighted",
        "roc_auc": "roc_auc",
    }

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    summary: dict[str, float] = {}
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        summary[f"cv_{metric}_mean"] = float(np.mean(scores))
        summary[f"cv_{metric}_std"] = float(np.std(scores))
        logger.info(
            "  CV %-18s  mean=%.4f  std=%.4f",
            metric + ":",
            np.mean(scores),
            np.std(scores),
        )

    return summary


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_pipeline(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate a fitted pipeline on the held-out test set.

    Parameters
    ----------
    pipeline:
        Fitted pipeline.
    X_test:
        Test feature matrix.
    y_test:
        Test target series.

    Returns
    -------
    dict
        Accuracy, F1 (weighted), and ROC-AUC on the test set.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # probability of class 1

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    logger.info("─" * 55)
    logger.info("TEST SET RESULTS")
    logger.info("  Accuracy (test)    : %.4f", metrics["test_accuracy"])
    logger.info("  F1-weighted (test) : %.4f", metrics["test_f1_weighted"])
    logger.info("  ROC-AUC (test)     : %.4f", metrics["test_roc_auc"])
    logger.info("─" * 55)

    logger.info("Classification Report:\n%s",
                classification_report(y_test, y_pred, target_names=["stayed", "quit"]))

    return metrics


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------


def log_feature_importances(
    pipeline: Pipeline,
    top_n: int = 15,
) -> pd.DataFrame:
    """Extract and log the top-N feature importances from the forest.

    Parameters
    ----------
    pipeline:
        Fitted pipeline containing a ColumnTransformer and a RandomForest.
    top_n:
        Number of top features to display.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'feature' and 'importance' columns, sorted descending.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Get feature names produced by the ColumnTransformer
    feature_names: list[str] = preprocessor.get_feature_names_out().tolist()
    importances: np.ndarray = classifier.feature_importances_

    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    logger.info("Top-%d Feature Importances:", top_n)
    for _, row in fi_df.head(top_n).iterrows():
        bar = "█" * int(row["importance"] * 50)
        logger.info("  %-45s %.4f  %s", row["feature"], row["importance"], bar)

    return fi_df


# ---------------------------------------------------------------------------
# Model Persistence
# ---------------------------------------------------------------------------


def save_pipeline(
    pipeline: Pipeline,
    output_dir: Path,
    filename: str = DEFAULT_MODEL_FILENAME,
) -> Path:
    """Serialize the fitted pipeline to disk using joblib.

    Parameters
    ----------
    pipeline:
        Fitted sklearn Pipeline.
    output_dir:
        Directory where the model file will be saved (created if missing).
    filename:
        Name of the output file.

    Returns
    -------
    Path
        Absolute path to the saved model file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / filename
    joblib.dump(pipeline, model_path, compress=3)
    logger.info("Pipeline saved → %s  (%.1f KB)", model_path.resolve(),
                model_path.stat().st_size / 1024)
    return model_path


def save_metrics(
    metrics: dict[str, float],
    output_dir: Path,
    filename: str = DEFAULT_METRICS_FILENAME,
) -> Path:
    """Save evaluation metrics as a JSON file.

    Parameters
    ----------
    metrics:
        Dictionary of metric name → value.
    output_dir:
        Directory for the JSON file.
    filename:
        Output filename.

    Returns
    -------
    Path
        Absolute path to the saved JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / filename
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved → %s", metrics_path.resolve())
    return metrics_path


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Employee Attrition Random Forest classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the employee_data.csv file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the trained model and metrics.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of data for the test set.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees in the Random Forest.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum tree depth (None = unlimited).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Employee Attrition — Model Training")
    logger.info("=" * 60)
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Load & split data
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = load_and_prepare(
        csv_path=args.data,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # ------------------------------------------------------------------
    # 2. Build pipeline
    # ------------------------------------------------------------------
    pipeline = build_pipeline(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )

    # ------------------------------------------------------------------
    # 3. Cross-validation (on train data only — no data leakage)
    # ------------------------------------------------------------------
    cv_metrics = cross_validate_pipeline(
        pipeline, X_train, y_train,
        n_splits=args.cv_folds,
        random_state=args.random_state,
    )

    # ------------------------------------------------------------------
    # 4. Final fit on full training set
    # ------------------------------------------------------------------
    logger.info("Fitting final model on full training set …")
    t_fit = time.perf_counter()
    pipeline.fit(X_train, y_train)
    logger.info("Training complete in %.2f seconds.", time.perf_counter() - t_fit)

    # ------------------------------------------------------------------
    # 5. Evaluate on held-out test set
    # ------------------------------------------------------------------
    test_metrics = evaluate_pipeline(pipeline, X_test, y_test)

    # ------------------------------------------------------------------
    # 6. Feature importances
    # ------------------------------------------------------------------
    fi_df = log_feature_importances(pipeline)

    # ------------------------------------------------------------------
    # 7. Save model + metrics
    # ------------------------------------------------------------------
    all_metrics = {**cv_metrics, **test_metrics}
    save_pipeline(pipeline, output_dir=args.output)
    save_metrics(all_metrics, output_dir=args.output)

    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info("Training pipeline finished in %.2f seconds.", elapsed)
    logger.info("Model: %s", args.output / DEFAULT_MODEL_FILENAME)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
