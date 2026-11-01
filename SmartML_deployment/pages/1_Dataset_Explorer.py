"""
SmartML: Dataset Input, Exploration & Task Configuration
Page 1: Dataset Explorer
"""

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Dataset Explorer - SmartML",
    page_icon="📊",
    layout="wide"
)

from utils.ui_styles import apply_custom_styles, render_header, render_sidebar_status, render_metric_card
from utils.state import init_session_state, set_new_dataset, invalidate_trained_models
from utils.data_loader import (
    load_uploaded_file,
    load_sample_dataset,
    get_dataset_summary,
    detect_prediction_task,
    validate_dataset_for_task
)


def main():
    apply_custom_styles()
    init_session_state()
    render_sidebar_status()

    render_header(
        title="Dataset Explorer & Preprocessing Setup",
        subtitle="Upload datasets, inspect distributions, review missing values, select target and feature columns, and configure the predictive task.",
        icon="📊"
    )

    # 1. Dataset Acquisition Section
    st.markdown("### 📥 1. Load Data")
    load_tab1, load_tab2 = st.tabs(["📁 Upload Your File (CSV / Excel)", "🧪 Use Benchmark Sample Datasets"])

    with load_tab1:
        uploaded_file = st.file_uploader(
            "Upload a tabular dataset (.csv, .xlsx, or .xls):",
            type=["csv", "xlsx", "xls"],
            key="file_uploader_widget"
        )
        if uploaded_file is not None:
            # Check if this is a newly uploaded file or current file
            if st.session_state.get("last_uploaded_name") != uploaded_file.name:
                df, err = load_uploaded_file(uploaded_file)
                if err:
                    st.error(f"Error loading file: {err}")
                elif df is not None:
                    st.session_state["last_uploaded_name"] = uploaded_file.name
                    set_new_dataset(df, f"Uploaded: {uploaded_file.name}")
                    st.success(f"Successfully loaded '{uploaded_file.name}' ({len(df)} rows, {len(df.columns)} columns)!")
                    st.rerun()

    with load_tab2:
        st.markdown("Select a pre-configured real-world machine learning benchmark dataset:")
        sample_choice = st.radio(
            "Choose Sample Dataset:",
            options=["Student Performance (Regression)", "Loan Approval Risk (Classification)"],
            horizontal=True
        )
        if st.button("Load Selected Benchmark", key="btn_load_sample_tab"):
            dtype = "regression" if "Regression" in sample_choice else "classification"
            sample_df, sample_target, sample_task, desc = load_sample_dataset(dtype)
            if sample_df is not None:
                set_new_dataset(sample_df, f"Sample: {sample_choice}", sample_target, sample_task)
                st.success(f"Loaded {sample_choice}! {desc}")
                st.rerun()

    df = st.session_state.get("df")
    if df is None:
        st.info("👆 Please upload a dataset or load one of the built-in benchmark datasets above to begin.")
        return

    st.markdown("---")

    # 2. Dataset Overview Metrics
    st.markdown(f"### 📋 2. Dataset Profile: `{st.session_state.get('dataset_name', 'Active')}`")
    summary = get_dataset_summary(df)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Total Observations", f"{summary['num_rows']:,}", "Number of rows", "Size", "blue")
    with m2:
        render_metric_card("Total Features", f"{summary['num_cols']:,}", f"{len(summary['numeric_cols'])} Num, {len(summary['categorical_cols'])} Cat", "Columns", "purple")
    with m3:
        miss_color = "green" if summary["missing_cells"] == 0 else "amber"
        render_metric_card("Missing Values", f"{summary['missing_cells']:,}", f"{summary['missing_pct']:.2f}% of all cells", "Nulls", miss_color)
    with m4:
        dup_color = "green" if summary["duplicates_count"] == 0 else "amber"
        render_metric_card("Duplicate Rows", f"{summary['duplicates_count']:,}", "Identical rows detected", "Duplicates", dup_color)

    # Preview Table
    with st.expander("🔍 Preview First 10 Rows of Dataset", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

    # Column Details Table
    with st.expander("📊 Detailed Column Information & Data Types", expanded=False):
        col_df = pd.DataFrame(summary["columns_info"])
        st.dataframe(col_df, use_container_width=True, hide_index=True)

    # Descriptive Statistics
    with st.expander("📈 Descriptive Statistics Summary", expanded=False):
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.markdown("---")

    # 3. Target and Feature Column Selection
    st.markdown("### 🎯 3. Select Target & Input Features")
    st.markdown("Define the variable you want to predict (Target) and the input variables used for learning (Features).")

    all_columns = list(df.columns)
    current_target = st.session_state.get("target_col")
    if current_target not in all_columns:
        current_target = all_columns[-1]

    col_target, col_features = st.columns([1, 2])

    with col_target:
        selected_target = st.selectbox(
            "Select Prediction Target Column:",
            options=all_columns,
            index=all_columns.index(current_target),
            help="The dependent variable (y) that your machine learning model will learn to predict."
        )

        # Invalidate models if target changed
        if selected_target != st.session_state.get("target_col"):
            st.session_state["target_col"] = selected_target
            invalidate_trained_models("Target column changed")

    # 4. Intelligent Task Detection & Selection
    rec_task, conf, task_explanation = detect_prediction_task(df, selected_target)
    current_task = st.session_state.get("task", rec_task)

    with col_target:
        st.markdown(
            f"""
            <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 0.8rem; margin-top: 0.8rem;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #1E40AF;">🤖 SmartML Task Recommendation:</div>
                <div style="font-size: 0.82rem; color: #1E3A8A; margin-top: 0.2rem;">
                    Recommended: <b>{rec_task.upper()}</b> ({conf.capitalize()} confidence)<br>
                    <span style="font-size: 0.78rem; color: #475569;">{task_explanation}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        selected_task = st.radio(
            "Confirm or Change Prediction Task:",
            options=["regression", "classification"],
            index=0 if current_task == "regression" else 1,
            format_func=lambda x: "📈 Regression (Predict continuous numerical quantity)" if x == "regression" else "🎯 Classification (Predict discrete class / category)",
            help="Choose Regression for continuous numbers (e.g. house price, test score). Choose Classification for categories or binary outcomes (e.g. loan approval, disease diagnosis)."
        )

        if selected_task != st.session_state.get("task"):
            st.session_state["task"] = selected_task
            invalidate_trained_models("Task changed")

    with col_features:
        # Candidate features are all columns except target
        candidate_features = [c for c in all_columns if c != selected_target]
        current_features = st.session_state.get("feature_cols", candidate_features)
        # Filter to only valid candidates
        valid_current_features = [c for c in current_features if c in candidate_features]
        if not valid_current_features:
            valid_current_features = candidate_features

        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button("Select All Features", use_container_width=True):
                st.session_state["feature_cols"] = candidate_features
                invalidate_trained_models("Features modified")
                st.rerun()
        with btn_c2:
            if st.button("Deselect All", use_container_width=True):
                st.session_state["feature_cols"] = []
                invalidate_trained_models("Features modified")
                st.rerun()

        selected_features = st.multiselect(
            "Select Input Predictor Features (X):",
            options=candidate_features,
            default=valid_current_features,
            help="Choose the attributes to feed into the algorithm. The target column is excluded automatically to prevent data leakage."
        )

        if selected_features != st.session_state.get("feature_cols"):
            st.session_state["feature_cols"] = selected_features
            invalidate_trained_models("Features modified")

        st.caption(f"Selected {len(selected_features)} of {len(candidate_features)} candidate features.")

    st.markdown("---")

    # 5. Dataset Validation & Next Steps
    st.markdown("### ✅ 4. Validation & Preprocessing Check")
    is_valid, errors, warnings = validate_dataset_for_task(
        df,
        selected_target,
        selected_features,
        selected_task
    )

    if errors:
        for err in errors:
            st.error(f"❌ {err}")
    else:
        st.success("✅ Dataset and feature selection are valid and ready for training!")

    if warnings:
        for warn in warnings:
            st.warning(f"⚠️ {warn}")

    if is_valid:
        st.markdown(
            """
            <div class="callout-box">
                <b>Ready to Proceed:</b> Your configuration has been saved.
                Proceed to the <b>Model Selection</b> page to choose an algorithm, inspect its compatibility, and train your model.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.page_link("pages/2_Model_Selection.py", label="Proceed to Model Selection →", icon="⚙️")


if __name__ == "__main__":
    main()
