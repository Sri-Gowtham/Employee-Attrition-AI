"""
data_preprocessing.py
---------------------
Handles all data loading, validation, and preprocessing for the
Employee Attrition Prediction project.

Responsibilities:
- Load CSV data with validation
- Define feature/target columns
- Build a reproducible sklearn ColumnTransformer preprocessing pipeline
- Provide a train/test split utility

Python 3.11+ required.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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
# Column Definitions
# ---------------------------------------------------------------------------

TARGET_COLUMN: str = "quit"

NUMERICAL_FEATURES: list[str] = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",   # original typo preserved to match CSV header
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
]

CATEGORICAL_FEATURES: list[str] = [
    "department",
    "salary",
]

ALL_FEATURES: list[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Salary has a natural ordering; we still use OHE for simplicity
# (upgrade path: use OrdinalEncoder with salary ordering)
EXPECTED_DEPARTMENTS: tuple[str, ...] = (
    "sales",
    "accounting",
    "hr",
    "technical",
    "support",
    "management",
    "IT",
    "product_mng",
    "marketing",
    "RandD",
)
EXPECTED_SALARY_LEVELS: tuple[str, ...] = ("low", "medium", "high")

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load employee data from a CSV file with basic validation.

    Parameters
    ----------
    csv_path:
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all original columns.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at the given path.
    ValueError
        If required columns are missing from the CSV.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}")

    logger.info("Loading data from: %s", path.resolve())
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns.", len(df), len(df.columns))

    _validate_columns(df)
    _validate_values(df)

    return df


def _validate_columns(df: pd.DataFrame) -> None:
    """Ensure all expected feature and target columns exist."""
    required = set(ALL_FEATURES + [TARGET_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}\n"
            f"Found columns: {sorted(df.columns.tolist())}"
        )
    logger.info("Column validation passed.")


def _validate_values(df: pd.DataFrame) -> None:
    """Perform lightweight sanity checks on the data."""
    # Check target values
    invalid_targets = set(df[TARGET_COLUMN].unique()) - {0, 1}
    if invalid_targets:
        raise ValueError(
            f"Unexpected values in target column '{TARGET_COLUMN}': {invalid_targets}"
        )

    # Check for unexpected categories (warning only — OHE handles unseen categories)
    unknown_depts = (
        set(df["department"].unique()) - set(EXPECTED_DEPARTMENTS)
    )
    if unknown_depts:
        logger.warning(
            "Unknown department values found (will be treated as unseen by OHE): %s",
            unknown_depts,
        )

    unknown_salary = (
        set(df["salary"].unique()) - set(EXPECTED_SALARY_LEVELS)
    )
    if unknown_salary:
        logger.warning(
            "Unknown salary values found: %s", unknown_salary
        )

    # Check for missing values
    null_counts = df[ALL_FEATURES + [TARGET_COLUMN]].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        logger.warning(
            "Missing values detected:\n%s\nConsider adding imputation to the pipeline.",
            cols_with_nulls.to_string(),
        )
    else:
        logger.info("No missing values detected.")


# ---------------------------------------------------------------------------
# Feature / Target Split
# ---------------------------------------------------------------------------


def get_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into feature matrix X and target series y.

    Parameters
    ----------
    df:
        Dataframe containing both features and target.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (ALL_FEATURES columns only).
    y : pd.Series
        Binary target series (0 = stayed, 1 = quit).
    """
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()
    logger.info(
        "Feature matrix: %d rows × %d cols | Target: %d rows",
        X.shape[0], X.shape[1], len(y),
    )
    return X, y


# ---------------------------------------------------------------------------
# Train / Test Split
# ---------------------------------------------------------------------------


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Target series.
    test_size:
        Fraction of data reserved for the test set (default 0.20).
    random_state:
        Seed for reproducibility.
    stratify:
        Whether to stratify the split on the target (recommended for
        imbalanced datasets).

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
    logger.info(
        "Train size: %d rows | Test size: %d rows (test_size=%.0f%%)",
        len(X_train), len(X_test), test_size * 100,
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Preprocessing Pipeline
# ---------------------------------------------------------------------------


def build_preprocessor() -> ColumnTransformer:
    """Build and return the sklearn ColumnTransformer preprocessor.

    Transformations applied:
    - Categorical columns  → fill missing with 'missing' string, then OneHotEncoder
                              (drops first to avoid multicollinearity,
                               handle_unknown='ignore' for unseen categories at inference)
    - Numerical columns    → fill missing with column median, then pass through
                              (tree models don't need scaling)

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor — fit it inside a Pipeline on training data only.
    """
    # Categorical branch: impute → OHE
    categorical_transformer = Pipeline(steps=[
        (
            "imputer",
            SimpleImputer(strategy="constant", fill_value="missing"),
        ),
        (
            "onehot",
            OneHotEncoder(
                drop="first",            # avoids dummy variable trap
                handle_unknown="ignore", # silently zeros out unseen categories at inference
                sparse_output=False,     # return a dense array (easier debugging)
                dtype=float,
            ),
        ),
    ])

    # Numerical branch: impute median → passthrough
    numerical_transformer = Pipeline(steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                categorical_transformer,
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                numerical_transformer,
                NUMERICAL_FEATURES,
            ),
        ],
        remainder="drop",        # drop any unexpected columns silently
        verbose_feature_names_out=True,
    )

    logger.info(
        "Preprocessor built — categorical: %s | numerical: %s",
        CATEGORICAL_FEATURES,
        NUMERICAL_FEATURES,
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Convenience: end-to-end load → split
# ---------------------------------------------------------------------------


def load_and_prepare(
    csv_path: str | Path,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """One-shot helper: load CSV → validate → split.

    Parameters
    ----------
    csv_path:
        Path to the employee data CSV.
    test_size:
        Fraction reserved for testing.
    random_state:
        RNG seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    df = load_data(csv_path)
    X, y = get_features_and_target(df)
    return split_data(X, y, test_size=test_size, random_state=random_state)
