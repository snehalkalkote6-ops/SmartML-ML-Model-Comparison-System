"""
SmartML: Model Compatibility, Training & Empirical Evaluation
Page 2: Model Selection
"""

import time
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Model Selection - SmartML",
    page_icon="⚙️",
    layout="wide"
)

from utils.ui_styles import (
    apply_custom_styles,
    render_header,
    render_sidebar_status,
    render_metric_card
)
from utils.state import init_session_state
from utils.data_loader import validate_dataset_for_task
from utils.preprocessing import SmartMLPipeline
from utils.model_registry import (
    get_model_catalog,
    check_model_compatibility,
    instantiate_model
)
from utils.evaluation import (
    evaluate_regression_model,
    evaluate_classification_model
)
from utils.visualization import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_error_distribution,
    plot_confusion_matrix,
    plot_roc_curves
)
from utils.recommendations import generate_single_model_recommendations


def main():
    apply_custom_styles()
    init_session_state()
    render_sidebar_status()

    render_header(
        title="Model Selection, Compatibility & Evaluation",
        subtitle="Select machine learning algorithms, verify empirical compatibility, train models with leak-free preprocessing, and inspect real performance metrics.",
        icon="⚙️"
    )

    df = st.session_state.get("df")
    target_col = st.session_state.get("target_col")
    feature_cols = st.session_state.get("feature_cols", [])
    task = st.session_state.get("task", "regression")

    # 1. Prerequisite Validation
    if df is None:
        st.warning("⚠️ No dataset has been loaded yet.")
        st.page_link("pages/1_Dataset_Explorer.py", label="Go to Dataset Explorer to load data →", icon="📊")
        return

    is_valid, errors, warnings = validate_dataset_for_task(df, target_col, feature_cols, task)
    if not is_valid:
        st.error("❌ Dataset configuration is currently invalid:")
        for err in errors:
            st.write(f"• {err}")
        st.page_link("pages/1_Dataset_Explorer.py", label="Fix configuration in Dataset Explorer →", icon="📊")
        return

    # 2. Model Catalog & Compatibility Inspection
    catalog = get_model_catalog(task)
    model_names = list(catalog.keys())

    st.markdown(f"### 🧠 1. Select Model for `{task.upper()}`")
    col_sel, col_info = st.columns([1, 2])

    with col_sel:
        selected_model_name = st.selectbox(
            "Choose Algorithm:",
            options=model_names,
            help="Select an algorithm to inspect its compatibility and train."
        )

        model_meta = catalog[selected_model_name]

        # Check compatibility
        valid_targets = df[target_col].dropna()
        n_classes = valid_targets.nunique() if task == "classification" else 0
        compat_info = check_model_compatibility(
            selected_model_name,
            task,
            num_features=len(feature_cols),
            num_samples=len(df),
            num_classes=n_classes
        )

        # Compatibility Status Badge
        badge_style = "badge-green" if compat_info["status"] == "Compatible" else ("badge-amber" if compat_info["status"] == "Warning" else "badge-red")
        st.markdown(
            f"""
            <div style="margin-top: 0.8rem; padding: 0.8rem; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px;">
                <div style="font-size: 0.78rem; font-weight: 700; color: #64748B;">COMPATIBILITY STATUS:</div>
                <div style="margin-top: 0.2rem;"><span class="smart-badge {badge_style}">{compat_info['status'].upper()}</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_info:
        st.markdown(
            f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <h3 style="margin: 0; color: #1E3A8A; font-size: 1.25rem;">{model_meta['name']}</h3>
                    <span class="smart-badge badge-blue">{model_meta['type']}</span>
                </div>
                <p style="margin: 0 0 0.8rem 0; color: #334155; font-size: 0.9rem; line-height: 1.5;">{model_meta['description']}</p>
                <div style="font-size: 0.84rem; color: #0F172A; line-height: 1.6;">
                    <b>💡 Best Suited For:</b> {model_meta['suitable_for']}<br>
                    <b>⚠️ Limitations & Assumptions:</b> {model_meta['limitations']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display compatibility reasons
        for reason in compat_info["reasons"]:
            if compat_info["is_compatible"]:
                st.info(f"ℹ️ {reason}")
            else:
                st.error(f"❌ {reason}")

        for sug in compat_info["suggestions"]:
            st.warning(f"💡 Recommendation: {sug}")

    st.markdown("---")

    # 3. Hyperparameters & Pipeline Training Configuration
    st.markdown("### ⚙️ 2. Training & Split Configuration")
    conf1, conf2, conf3 = st.columns(3)

    with conf1:
        test_size = st.slider(
            "Test Set Proportion (Hold-out):",
            min_value=0.10,
            max_value=0.40,
            value=0.25,
            step=0.05,
            help="Proportion of the dataset reserved exclusively for unbiased test evaluation."
        )

    with conf2:
        random_seed = st.number_input(
            "Random Split Seed:",
            min_value=1,
            max_value=9999,
            value=42,
            step=1,
            help="Ensures exact reproducibility of data splits and training."
        )

    # Dynamic Hyperparameter controls based on selected model
    custom_params = {}
    with conf3:
        if selected_model_name == "Polynomial Regression":
            poly_deg = st.selectbox("Polynomial Degree:", options=[2, 3], index=0)
            custom_params["degree"] = poly_deg
        elif selected_model_name == "Ridge Regression":
            alpha = st.slider("L2 Regularization Alpha:", min_value=0.01, max_value=20.0, value=1.0, step=0.1)
            custom_params["alpha"] = alpha
        elif selected_model_name == "k-Nearest Neighbors (k-NN)":
            k_val = st.slider("Number of Neighbors (k):", min_value=1, max_value=15, value=5, step=1)
            custom_params["n_neighbors"] = k_val
        elif selected_model_name == "Support Vector Machine (SVM)":
            c_val = st.slider("Regularization Parameter (C):", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            kernel = st.selectbox("Kernel Function:", options=["rbf", "linear"], index=0)
            custom_params["C"] = c_val
            custom_params["kernel"] = kernel
        elif selected_model_name == "Logistic Regression":
            c_val = st.slider("Inverse Regularization Strength (C):", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            custom_params["C"] = c_val
        elif "Random Forest" in selected_model_name:
            n_est = st.slider("Number of Trees (n_estimators):", min_value=20, max_value=200, value=100, step=10)
            depth = st.slider("Max Tree Depth:", min_value=2, max_value=20, value=8, step=1)
            custom_params["n_estimators"] = n_est
            custom_params["max_depth"] = depth
        else:
            st.info("No custom tuning required for this model baseline.")

    # 4. Training Trigger Button
    train_disabled = not compat_info["is_compatible"]
    if st.button("🚀 Train Selected Model", disabled=train_disabled, use_container_width=True, type="primary"):
        with st.spinner(f"Fitting leak-free preprocessing and training {selected_model_name}..."):
            start_time = time.time()
            try:
                # 1. Construct pipeline and fit preprocessor strictly on train split
                pipeline = SmartMLPipeline(
                    target_col=target_col,
                    feature_cols=feature_cols,
                    task=task,
                    test_size=test_size,
                    random_state=random_seed,
                    scale_numerical=True
                )
                X_train_proc, X_test_proc, y_train, y_test, meta = pipeline.fit_and_split(df)

                # 2. Instantiate and train model
                model = instantiate_model(selected_model_name, task, custom_params)
                model.fit(X_train_proc, y_train)
                elapsed_time = time.time() - start_time

                # 3. Evaluate model
                if task == "regression":
                    eval_results = evaluate_regression_model(
                        model,
                        X_train_proc,
                        y_train,
                        X_test_proc,
                        y_test
                    )
                else:
                    eval_results = evaluate_classification_model(
                        model,
                        X_train_proc,
                        y_train,
                        X_test_proc,
                        y_test,
                        class_labels=pipeline.target_classes
                    )

                eval_results["training_time_sec"] = elapsed_time
                eval_results["sample_metadata"] = meta

                # 4. Save to session state
                st.session_state["pipeline"] = pipeline
                st.session_state["fitted_model"] = model
                st.session_state["trained_model_name"] = selected_model_name
                st.session_state["evaluation_results"] = eval_results
                st.session_state["is_model_trained"] = True

                st.success(f"🎉 Successfully trained **{selected_model_name}** in {elapsed_time:.3f} seconds!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")

    st.markdown("---")

    # 5. Display Evaluation Results
    is_trained = st.session_state.get("is_model_trained", False)
    eval_res = st.session_state.get("evaluation_results")
    active_model = st.session_state.get("trained_model_name")

    if not is_trained or eval_res is None:
        st.info("👆 Click **'Train Selected Model'** above to compute empirical evaluation metrics and visualizations.")
        return

    st.markdown(f"### 📈 3. Evaluation Results: `{active_model}`")
    st.caption(f"Trained in {eval_res.get('training_time_sec', 0.0):.3f}s &bull; Tested on {eval_res.get('sample_metadata', {}).get('num_test_samples', 0)} held-out samples &bull; Transformed features: {eval_res.get('sample_metadata', {}).get('transformed_feature_count', 0)}")

    test_metrics = eval_res.get("test_metrics", {})
    train_metrics = eval_res.get("train_metrics", {})

    if task == "regression":
        # Metric Cards for Regression
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            render_metric_card("R² Score (Test)", f"{test_metrics['r2']:.4f}", f"Train R²: {train_metrics['r2']:.4f}", "Variance Explained", "green" if test_metrics['r2'] > 0.7 else "amber")
        with rc2:
            render_metric_card("RMSE", f"{test_metrics['rmse']:.4f}", f"Train RMSE: {train_metrics['rmse']:.4f}", "Root Mean Sq Err", "blue")
        with rc3:
            render_metric_card("MAE", f"{test_metrics['mae']:.4f}", f"Train MAE: {train_metrics['mae']:.4f}", "Mean Absolute Err", "blue")
        with rc4:
            render_metric_card("MSE", f"{test_metrics['mse']:.4f}", f"Train MSE: {train_metrics['mse']:.4f}", "Mean Squared Err", "purple")

        # Regression Visualizations
        st.markdown("#### 📊 Diagnostic Plots")
        v_tab1, v_tab2, v_tab3 = st.tabs(["Actual vs. Predicted", "Residual Plot", "Error Distribution"])

        with v_tab1:
            fig_act = plot_actual_vs_predicted(
                eval_res["predictions"]["y_test_actual"],
                eval_res["predictions"]["y_test_pred"],
                model_name=active_model
            )
            st.plotly_chart(fig_act, use_container_width=True)

        with v_tab2:
            fig_res = plot_residuals(
                eval_res["predictions"]["y_test_pred"],
                eval_res["predictions"]["residuals"],
                model_name=active_model
            )
            st.plotly_chart(fig_res, use_container_width=True)

        with v_tab3:
            fig_dist = plot_error_distribution(
                eval_res["predictions"]["residuals"],
                model_name=active_model
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    else:  # classification
        # Metric Cards for Classification
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            render_metric_card("Accuracy", f"{test_metrics['accuracy'] * 100:.1f}%", f"Train Acc: {train_metrics['accuracy'] * 100:.1f}%", "Overall Correct", "green" if test_metrics['accuracy'] > 0.8 else "amber")
        with cc2:
            render_metric_card("F1-Score (Weighted)", f"{test_metrics['f1_weighted']:.4f}", f"Macro F1: {test_metrics['f1_macro']:.4f}", "Harmonic Balance", "blue")
        with cc3:
            render_metric_card("Precision (Weighted)", f"{test_metrics['precision_weighted']:.4f}", f"Macro: {test_metrics['precision_macro']:.4f}", "Exactness", "purple")
        with cc4:
            roc_str = f"{test_metrics['roc_auc']:.4f}" if test_metrics.get("roc_auc") is not None else "N/A"
            render_metric_card("ROC-AUC", roc_str, "Area Under Curve", "Discrimination", "blue" if test_metrics.get("roc_auc") else "amber")

        # Classification Visualizations
        st.markdown("#### 📊 Diagnostic Plots")
        c_tab1, c_tab2 = st.tabs(["Confusion Matrix Heatmap", "ROC Curve Analysis"])

        with c_tab1:
            fig_cm = plot_confusion_matrix(
                eval_res["confusion_matrix"],
                class_labels=eval_res["class_labels"],
                model_name=active_model
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with c_tab2:
            if eval_res.get("roc_data"):
                fig_roc = plot_roc_curves(eval_res["roc_data"], model_name=active_model)
                if fig_roc:
                    st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("ROC curves require probability estimates. This algorithm does not provide calibrated probabilities or only 1 class is present.")

    st.markdown("---")

    # 6. Diagnostic Recommendations & Actionable Insights
    st.markdown("### 🔍 4. Intelligent Diagnostic Analysis")
    diagnostics = generate_single_model_recommendations(
        active_model,
        task,
        eval_res,
        eval_res.get("sample_metadata", {})
    )

    diag_c1, diag_c2 = st.columns(2)

    with diag_c1:
        st.markdown("#### 📝 Empirical Observations:")
        for ins in diagnostics["insights"]:
            st.markdown(f"• {ins}", unsafe_allow_html=True)

        if diagnostics.get("overfitting_alert"):
            alert = diagnostics["overfitting_alert"]
            st.warning(f"⚠️ **{alert['type']} ({alert['severity']} Severity):** {alert['message']}")

    with diag_c2:
        st.markdown("#### 🛠️ Recommended Next Steps:")
        for step in diagnostics["next_steps"]:
            st.markdown(f"• {step}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Next Steps Navigation
    nxt_c1, nxt_c2 = st.columns(2)
    with nxt_c1:
        st.page_link("pages/3_Model_Comparison.py", label="Compare All Compatible Models →", icon="📈")
    with nxt_c2:
        st.page_link("pages/4_Make_Predictions.py", label="Make Predictions on New Data →", icon="🔮")


if __name__ == "__main__":
    main()
