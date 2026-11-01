"""
SmartML State Management Helper
Ensures consistent session state initialization and invalidation across all pages.
"""

from typing import List, Optional
import streamlit as st
import pandas as pd


def init_session_state():
    """
    Initializes default session state keys if not already present.
    """
    defaults = {
        "df": None,
        "dataset_name": "None",
        "target_col": None,
        "feature_cols": [],
        "task": "regression",
        "pipeline": None,
        "fitted_model": None,
        "trained_model_name": None,
        "evaluation_results": None,
        "is_model_trained": False,
        "comparison_results": None,
        "comparison_df": None,
        "test_size": 0.25,
        "random_state": 42
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def invalidate_trained_models(reason: str = ""):
    """
    Clears any trained model or comparison results when dataset or features change.
    """
    st.session_state["pipeline"] = None
    st.session_state["fitted_model"] = None
    st.session_state["trained_model_name"] = None
    st.session_state["evaluation_results"] = None
    st.session_state["is_model_trained"] = False
    st.session_state["comparison_results"] = None
    st.session_state["comparison_df"] = None


def set_new_dataset(df: pd.DataFrame, dataset_name: str, default_target: Optional[str] = None, default_task: Optional[str] = None):
    """
    Sets a newly loaded dataset and cleans up stale models.
    """
    invalidate_trained_models("New dataset loaded")
    st.session_state["df"] = df
    st.session_state["dataset_name"] = dataset_name

    # Set default target if valid
    if default_target and default_target in df.columns:
        st.session_state["target_col"] = default_target
        st.session_state["feature_cols"] = [c for c in df.columns if c != default_target]
    else:
        # Default to the last column
        st.session_state["target_col"] = df.columns[-1]
        st.session_state["feature_cols"] = list(df.columns[:-1])

    if default_task:
        st.session_state["task"] = default_task
