"""
SmartML: Live Single-Sample Inference & Predictions
Page 4: Make Predictions
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Make Predictions - SmartML",
    page_icon="🔮",
    layout="wide"
)

from utils.ui_styles import (
    apply_custom_styles,
    render_header,
    render_sidebar_status,
    render_metric_card
)
from utils.state import init_session_state
from utils.visualization import plot_prediction_probabilities


def main():
    apply_custom_styles()
    init_session_state()
    render_sidebar_status()

    render_header(
        title="Live Sample Prediction & Inference",
        subtitle="Supply custom feature values for a new, unseen record to perform live inference using your fitted preprocessing pipeline and trained algorithm.",
        icon="🔮"
    )

    pipeline = st.session_state.get("pipeline")
    fitted_model = st.session_state.get("fitted_model")
    trained_model_name = st.session_state.get("trained_model_name")
    is_trained = st.session_state.get("is_model_trained", False)
    df = st.session_state.get("df")
    task = st.session_state.get("task", "regression")
    target_col = st.session_state.get("target_col")
    feature_cols = st.session_state.get("feature_cols", [])

    # Guard: Ensure model is trained
    if not is_trained or fitted_model is None or pipeline is None or df is None:
        st.warning("⚠️ No trained model is currently available in the session.")
        st.markdown("Before making live predictions, you must train an algorithm on your selected dataset.")
        st.page_link("pages/2_Model_Selection.py", label="Train a Model in Model Selection →", icon="⚙️")
        return

    st.markdown(f"### 🤖 Active Inference Model: `{trained_model_name}`")
    st.caption(f"Task: {task.upper()} &bull; Target: '{target_col}' &bull; Required Predictor Features: {len(feature_cols)}")

    st.markdown("---")

    # Dynamic Input Form Generator
    st.markdown("### 📝 Enter Values for New Sample")
    st.markdown("Input fields have been automatically configured to match the types and ranges observed in your dataset:")

    user_inputs = {}

    # Organize inputs into responsive 2-column or 3-column grid
    cols_per_row = 3 if len(feature_cols) >= 3 else len(feature_cols)
    if cols_per_row == 0:
        cols_per_row = 1

    grid_cols = st.columns(cols_per_row)

    for i, col_name in enumerate(feature_cols):
        grid_target = grid_cols[i % cols_per_row]
        series = df[col_name].dropna()

        with grid_target:
            if pd.api.types.is_numeric_dtype(series):
                # Numerical input
                min_val = float(series.min())
                max_val = float(series.max())
                median_val = float(series.median())
                std_val = float(series.std()) if len(series) > 1 else 1.0

                step_val = 1.0 if pd.api.types.is_integer_dtype(series) else round(std_val / 10.0, 3)
                if step_val <= 0:
                    step_val = 0.1

                val = st.number_input(
                    f"{col_name}:",
                    value=median_val,
                    step=step_val,
                    help=f"Observed dataset range: [{min_val:.2f}, {max_val:.2f}] (Median: {median_val:.2f})"
                )
                user_inputs[col_name] = val
            else:
                # Categorical selectbox
                unique_cats = sorted(series.unique().tolist())
                val = st.selectbox(
                    f"{col_name}:",
                    options=unique_cats,
                    index=0,
                    help=f"Select from observed categories: {', '.join([str(c) for c in unique_cats[:5]])}"
                )
                user_inputs[col_name] = val

    st.markdown("<br>", unsafe_allow_html=True)

    # Prediction Action Button
    if st.button("🔮 Compute Prediction", type="primary", use_container_width=True):
        try:
            # 1. Transform input record through the exact fitted ColumnTransformer
            X_input_proc = pipeline.transform_single(user_inputs)

            # 2. Compute prediction
            raw_pred = fitted_model.predict(X_input_proc)

            st.markdown("---")
            st.markdown("### 🎯 Inference Results")

            if task == "regression":
                pred_val = float(raw_pred[0])
                c_res1, c_res2 = st.columns([1, 2])
                with c_res1:
                    render_metric_card(
                        title=f"Predicted {target_col}",
                        value=f"{pred_val:.3f}",
                        subtext=f"Computed by {trained_model_name}",
                        badge_text="Continuous",
                        badge_color="green"
                    )
                with c_res2:
                    st.markdown(
                        f"""
                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.2rem;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #1E3A8A;">Prediction Summary</h4>
                            <p style="margin: 0; color: #334155; font-size: 0.92rem; line-height: 1.6;">
                                The model predicts that for an observation with the provided feature profile,
                                the expected quantitative value of <b>'{target_col}'</b> is approximately <b>{pred_val:.3f}</b>.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:  # classification
                decoded_class = pipeline.decode_target(raw_pred[0])
                has_proba = hasattr(fitted_model, "predict_proba")

                c_res1, c_res2 = st.columns([1, 2])
                with c_res1:
                    render_metric_card(
                        title=f"Predicted Class",
                        value=str(decoded_class),
                        subtext=f"Target: '{target_col}'",
                        badge_text="Discrete Class",
                        badge_color="blue"
                    )

                with c_res2:
                    st.markdown(
                        f"""
                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.2rem;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #1E3A8A;">Classification Output</h4>
                            <p style="margin: 0; color: #334155; font-size: 0.92rem; line-height: 1.6;">
                                Based on the decision boundary established during training,
                                this sample is classified as: <b style="color: #2563EB; font-size: 1.05rem;">'{decoded_class}'</b>.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Probabilities chart if supported
                if has_proba:
                    probs = fitted_model.predict_proba(X_input_proc)[0]
                    classes = pipeline.target_classes or [f"Class {i}" for i in range(len(probs))]

                    st.markdown("#### 📊 Confidence Probability Distribution")
                    fig_prob = plot_prediction_probabilities(probs.tolist(), classes)
                    st.plotly_chart(fig_prob, use_container_width=True)

            # Display formatted input record
            with st.expander("🔍 View Processed Input Vector & Record Details"):
                st.write("**User-supplied raw features:**")
                st.json(user_inputs)
                st.write(f"**Transformed feature array shape:** `{X_input_proc.shape}`")
                st.write(f"**Pipeline feature names:** `{pipeline.transformed_feature_names}`")

        except Exception as pred_err:
            st.error(f"❌ Prediction error: {str(pred_err)}")


if __name__ == "__main__":
    main()
