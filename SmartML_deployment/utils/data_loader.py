"""
SmartML Data Loader & Inspection Engine
Provides robust loading for CSV/Excel files, sample dataset integration,
comprehensive data profiling, automatic task recommendation, and dataset validation.
"""

import os
import io
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_sample_dataset(dataset_type: str = "regression") -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str], Optional[str]]:
    """
    Loads a built-in benchmark dataset.
    Returns: (df, target_col, recommended_task, description)
    """
    if dataset_type == "regression":
        file_path = os.path.join(SAMPLE_DATA_DIR, "sample_regression.csv")
        desc = "Student Academic Performance dataset (Predicting continuous Final Score from study hours, attendance, sleep, etc.)"
        target = "final_score"
        task = "regression"
    else:
        file_path = os.path.join(SAMPLE_DATA_DIR, "sample_classification.csv")
        desc = "Loan Approval Risk dataset (Predicting binary Loan Approval status from income, credit score, debt ratio, etc.)"
        target = "loan_approved"
        task = "classification"

    if not os.path.exists(file_path):
        return None, None, None, f"Sample dataset file not found at: {file_path}"

    try:
        df = pd.read_csv(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        return df, target, task, desc
    except Exception as e:
        return None, None, None, f"Error reading sample dataset: {str(e)}"


def load_uploaded_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Reads an uploaded CSV or Excel file safely with multi-encoding fallback.
    Returns: (df, error_message)
    """
    if uploaded_file is None:
        return None, "No file uploaded."

    filename = uploaded_file.name.lower()
    try:
        if filename.endswith(".csv"):
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
            except (UnicodeDecodeError, Exception):
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin1")
        elif filename.endswith((".xlsx", ".xls")):
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
        else:
            return None, f"Unsupported file format '{filename}'. Please upload a .csv, .xlsx, or .xls file."

        if df.empty:
            return None, "The uploaded dataset is empty (0 rows)."

        # Clean column headers
        df.columns = [str(col).strip() for col in df.columns]

        # Check for empty column names or duplicate column names
        cols = list(df.columns)
        if len(cols) != len(set(cols)):
            # Handle duplicate column names by renaming
            seen = {}
            new_cols = []
            for c in cols:
                if c in seen:
                    seen[c] += 1
                    new_cols.append(f"{c}_{seen[c]}")
                else:
                    seen[c] = 0
                    new_cols.append(c)
            df.columns = new_cols

        return df, None

    except Exception as e:
        return None, f"Failed to parse file: {str(e)}"


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates comprehensive dataset profiling metrics.
    """
    num_rows, num_cols = df.shape
    missing_cells = int(df.isna().sum().sum())
    total_cells = num_rows * num_cols
    missing_pct = (missing_cells / total_cells * 100.0) if total_cells > 0 else 0.0
    duplicates_count = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    columns_info = []
    constant_cols = []

    for col in df.columns:
        dtype_str = str(df[col].dtype)
        n_missing = int(df[col].isna().sum())
        col_miss_pct = (n_missing / num_rows * 100.0) if num_rows > 0 else 0.0
        n_unique = int(df[col].nunique(dropna=True))

        if n_unique <= 1:
            constant_cols.append(col)

        # Get readable sample unique values
        unique_samples = df[col].dropna().unique()[:3].tolist()
        sample_str = ", ".join([str(v) for v in unique_samples])
        if len(unique_samples) < df[col].nunique(dropna=True):
            sample_str += ", ..."

        columns_info.append({
            "Column": col,
            "Type": dtype_str,
            "Role": "Numerical" if col in numeric_cols else "Categorical",
            "Missing Count": n_missing,
            "Missing %": f"{col_miss_pct:.1f}%",
            "Unique Values": n_unique,
            "Samples": sample_str
        })

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
        "duplicates_count": duplicates_count,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "constant_cols": constant_cols,
        "columns_info": columns_info,
        "memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
    }


def detect_prediction_task(df: pd.DataFrame, target_col: str) -> Tuple[str, str, str]:
    """
    Intelligently analyzes target column to recommend task:
    Returns: (recommended_task, confidence, explanation)
    """
    if target_col not in df.columns:
        return "regression", "low", "Target column not found in dataset."

    series = df[target_col].dropna()
    if len(series) == 0:
        return "regression", "low", "Target column contains only missing values."

    n_unique = series.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(series)

    if not is_numeric:
        # String, object, category, or bool
        return (
            "classification",
            "high",
            f"The target column '{target_col}' has non-numeric categorical data ({n_unique} unique classes). "
            f"Classification is recommended to predict discrete class labels."
        )

    # Numeric target
    if n_unique <= 2:
        return (
            "classification",
            "high",
            f"The target column '{target_col}' is numeric but contains only {n_unique} distinct values "
            f"(typically binary 0/1 indicator). Classification is recommended."
        )
    elif n_unique <= 10 or (n_unique <= 15 and (n_unique / len(series) < 0.05)):
        return (
            "classification",
            "medium",
            f"The target column '{target_col}' has only {n_unique} distinct numeric levels. "
            f"This often represents ordinal or categorical codes (Classification). "
            f"However, if the numeric scale has continuous interval meaning, you may switch to Regression."
        )
    else:
        return (
            "regression",
            "high",
            f"The target column '{target_col}' contains continuous numeric values with {n_unique} distinct points. "
            f"Regression is recommended to predict quantitative numerical values."
        )


def validate_dataset_for_task(
    df: pd.DataFrame,
    target_col: Optional[str],
    feature_cols: List[str],
    task: str
) -> Tuple[bool, List[str], List[str]]:
    """
    Validates dataset, target, and feature selection before pipeline execution.
    Returns: (is_valid, error_messages, warning_messages)
    """
    errors = []
    warnings = []

    if df is None or df.empty:
        return False, ["Dataset is empty or not loaded."], []

    if df.shape[0] < 10:
        errors.append(f"Dataset has only {df.shape[0]} rows. At least 10 rows are required for train-test splitting.")

    if not target_col:
        errors.append("No target column has been selected.")
    elif target_col not in df.columns:
        errors.append(f"Target column '{target_col}' does not exist in the dataset.")
    else:
        # Check target missing values
        valid_targets = df[target_col].dropna()
        target_missing = len(df) - len(valid_targets)
        if len(valid_targets) < 5:
            errors.append(f"Target column '{target_col}' contains fewer than 5 non-missing values.")
        elif target_missing > 0:
            warnings.append(
                f"Target column '{target_col}' contains {target_missing} missing value(s). "
                f"These rows will be removed before training to prevent target leakage."
            )

        # Check unique target values for task
        n_unique_target = valid_targets.nunique()
        if n_unique_target < 2:
            errors.append(
                f"Target column '{target_col}' has only {n_unique_target} unique value. "
                f"Machine learning models require at least 2 distinct target values to learn."
            )

        if task == "classification":
            # Check class distributions
            class_counts = valid_targets.value_counts()
            min_class_count = class_counts.min()
            if min_class_count < 2:
                warnings.append(
                    f"Class '{class_counts.idxmin()}' appears only {min_class_count} time(s). "
                    f"Stratified splitting cannot be applied, so a standard random split will be used."
                )

            # Check if user mistakenly picked high-cardinality float for classification
            if pd.api.types.is_float_dtype(valid_targets) and n_unique_target > 50:
                warnings.append(
                    f"Target column has {n_unique_target} continuous float values for a Classification task. "
                    f"If you are predicting a continuous quantity, consider switching to Regression."
                )

        elif task == "regression":
            if not pd.api.types.is_numeric_dtype(valid_targets):
                errors.append(
                    f"Target column '{target_col}' contains non-numeric text values. "
                    f"Regression requires a continuous numerical target. Please select Classification or choose a numeric target."
                )
            elif n_unique_target <= 2:
                warnings.append(
                    f"Target column '{target_col}' has only {n_unique_target} unique values. "
                    f"This looks like binary data. Classification is usually more suitable for binary targets."
                )

    # Feature validation
    if not feature_cols:
        errors.append("No input features have been selected. Please select at least one feature column.")
    else:
        if target_col in feature_cols:
            errors.append(f"Target column '{target_col}' cannot also be an input feature (target leakage).")

        for f in feature_cols:
            if f not in df.columns:
                errors.append(f"Feature '{f}' not found in dataset columns.")
            else:
                n_uniq = df[f].nunique(dropna=True)
                if n_uniq <= 1:
                    warnings.append(f"Feature '{f}' has constant value (no variance). It will not contribute predictive signal.")

    # Check sample size warnings
    if df.shape[0] < 30:
        warnings.append(
            f"Small dataset alert: Total rows ({df.shape[0]}) is small. "
            f"Held-out test evaluations may have high statistical variance."
        )

    is_valid = len(errors) == 0
    return is_valid, errors, warnings
