"""
SmartML: Automatic Multi-Model Comparison & Benchmarking
Page 3: Model Comparison
"""

import time
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Model Comparison - SmartML",
    page_icon="📈",
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
    plot_model_comparison_bar,
    plot_train_vs_test_gap
)
from utils.recommendations import generate_comparison_recommendations


def main():
    apply_custom_styles()
    init_session_state()
    render_sidebar_status()

    render_header(
        title="Automated Multi-Model Comparison",
        subtitle="Benchmark all compatible machine learning models simultaneously on identical held-out validation splits to identify optimal trade-offs and performance.",
        icon="📈"
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

    catalog = get_model_catalog(task)
    all_models = list(catalog.keys())

    st.markdown(f"### 🎯 Benchmark Settings for `{task.upper()}`")
    st.markdown(f"SmartML will evaluate candidate algorithms (**{len(all_models)} available**) on the exact same train/test split to guarantee fair, leak-free benchmarking.")

    c1, c2 = st.columns(2)
    with c1:
        test_size = st.slider(
            "Validation Test Size (Hold-out):",
            min_value=0.10,
            max_value=0.40,
            value=0.25,
            step=0.05,
            key="comp_test_size"
        )
    with c2:
        random_seed = st.number_input(
            "Evaluation Random Seed:",
            min_value=1,
            max_value=9999,
            value=42,
            step=1,
            key="comp_seed"
        )

    # 2. Run Comparison Trigger
    if st.button("⚡ Run Multi-Model Comparison Benchmark", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        valid_targets = df[target_col].dropna()
        n_classes = valid_targets.nunique() if task == "classification" else 0

        # Create one unified leak-free train/test split for ALL models
        try:
            status_text.text("Splitting dataset and fitting leak-free preprocessing...")
            pipeline = SmartMLPipeline(
                target_col=target_col,
                feature_cols=feature_cols,
                task=task,
                test_size=test_size,
                random_state=random_seed,
                scale_numerical=True
            )
            X_train_proc, X_test_proc, y_train, y_test, meta = pipeline.fit_and_split(df)

            results_list = []
            total_models = len(all_models)

            for idx, model_name in enumerate(all_models):
                status_text.text(f"Evaluating {model_name} ({idx+1}/{total_models})...")
                progress_bar.progress((idx) / total_models)

                # Check model compatibility
                compat = check_model_compatibility(
                    model_name,
                    task,
                    num_features=len(feature_cols),
                    num_samples=len(df),
                    num_classes=n_classes
                )

                if not compat["is_compatible"]:
                    # Model is incompatible; record reason and continue gracefully
                    results_list.append({
                        "Model": model_name,
                        "Status": "Skipped (Incompatible)",
                        "Note": "; ".join(compat["reasons"]),
                        "Training Time (s)": np.nan,
                        "is_evaluated": False
                    })
                    continue

                # Instantiate and train model
                try:
                    start_t = time.time()
                    model_obj = instantiate_model(model_name, task)
                    model_obj.fit(X_train_proc, y_train)
                    train_duration = time.time() - start_t

                    if task == "regression":
                        ev = evaluate_regression_model(model_obj, X_train_proc, y_train, X_test_proc, y_test)
                        t_m = ev["test_metrics"]
                        tr_m = ev["train_metrics"]
                        results_list.append({
                            "Model": model_name,
                            "Status": "Evaluated",
                            "R² Score (Test)": round(t_m["r2"], 4),
                            "RMSE (Test)": round(t_m["rmse"], 4),
                            "MAE (Test)": round(t_m["mae"], 4),
                            "MSE (Test)": round(t_m["mse"], 4),
                            "R² Score (Train)": round(tr_m["r2"], 4),
                            "RMSE (Train)": round(tr_m["rmse"], 4),
                            "Training Time (s)": round(train_duration, 4),
                            "is_evaluated": True,
                            "model_obj": model_obj
                        })
                    else:
                        ev = evaluate_classification_model(model_obj, X_train_proc, y_train, X_test_proc, y_test, class_labels=pipeline.target_classes)
                        t_m = ev["test_metrics"]
                        tr_m = ev["train_metrics"]
                        roc_val = round(t_m["roc_auc"], 4) if t_m.get("roc_auc") is not None else np.nan
                        results_list.append({
                            "Model": model_name,
                            "Status": "Evaluated",
                            "Accuracy (Test)": round(t_m["accuracy"], 4),
                            "F1-Score (Test)": round(t_m["f1_weighted"], 4),
                            "Precision (Test)": round(t_m["precision_weighted"], 4),
                            "Recall (Test)": round(t_m["recall_weighted"], 4),
                            "ROC-AUC": roc_val,
                            "Accuracy (Train)": round(tr_m["accuracy"], 4),
                            "F1-Score (Train)": round(tr_m["f1_weighted"], 4),
                            "Training Time (s)": round(train_duration, 4),
                            "is_evaluated": True,
                            "model_obj": model_obj
                        })

                except Exception as model_err:
                    results_list.append({
                        "Model": model_name,
                        "Status": f"Failed: {str(model_err)}",
                        "Note": str(model_err),
                        "Training Time (s)": np.nan,
                        "is_evaluated": False
                    })

            progress_bar.progress(1.0)
            status_text.text("Comparison benchmark complete!")

            # Store comparison results
            st.session_state["comparison_results"] = results_list
            comp_df = pd.DataFrame(results_list)
            st.session_state["comparison_df"] = comp_df

            # If user has not yet trained a single model, set the top model as active
            eval_only_df = comp_df[comp_df["is_evaluated"] == True]
            if not eval_only_df.empty:
                if task == "regression":
                    top_row = eval_only_df.sort_values("RMSE (Test)", ascending=True).iloc[0]
                else:
                    top_row = eval_only_df.sort_values("F1-Score (Test)", ascending=False).iloc[0]

                top_model_name = top_row["Model"]
                st.session_state["pipeline"] = pipeline
                st.session_state["fitted_model"] = top_row["model_obj"]
                st.session_state["trained_model_name"] = top_model_name
                st.session_state["is_model_trained"] = True

            st.success("Comparison completed successfully!")
            st.rerun()

        except Exception as split_err:
            st.error(f"Failed to run benchmark: {str(split_err)}")

    st.markdown("---")

    # 3. Render Comparison Results
    comp_df = st.session_state.get("comparison_df")
    if comp_df is None or comp_df.empty:
        st.info("👆 Click **'Run Multi-Model Comparison Benchmark'** above to test and rank all models.")
        return

    st.markdown("### 📊 Benchmark Results Table")

    # Clean display dataframe (exclude internal object references)
    display_cols = [c for c in comp_df.columns if c not in ["model_obj", "is_evaluated"]]
    display_df = comp_df[display_cols].copy()

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    evaluated_df = comp_df[comp_df["is_evaluated"] == True].copy()
    if evaluated_df.empty:
        st.warning("No models were successfully evaluated.")
        return

    st.markdown("---")

    # 4. Interactive Metric Ranking & Chart
    st.markdown("### 🏆 Algorithm Ranking & Comparative Charts")

    if task == "regression":
        metric_opts = {
            "RMSE (Test)": {"higher_is_better": False, "desc": "Root Mean Squared Error (Lower is better)"},
            "MAE (Test)": {"higher_is_better": False, "desc": "Mean Absolute Error (Lower is better)"},
            "R² Score (Test)": {"higher_is_better": True, "desc": "R² Variance Explained (Higher is better)"}
        }
        default_metric = "RMSE (Test)"
    else:
        metric_opts = {
            "F1-Score (Test)": {"higher_is_better": True, "desc": "Weighted F1-Score (Higher is better)"},
            "Accuracy (Test)": {"higher_is_better": True, "desc": "Overall Accuracy (Higher is better)"},
            "Precision (Test)": {"higher_is_better": True, "desc": "Precision (Higher is better)"},
            "Recall (Test)": {"higher_is_better": True, "desc": "Recall (Higher is better)"}
        }
        default_metric = "F1-Score (Test)"

    col_rank_ctrl, col_gap_ctrl = st.columns(2)
    with col_rank_ctrl:
        chosen_metric = st.selectbox(
            "Rank Models By Criterion:",
            options=list(metric_opts.keys()),
            index=list(metric_opts.keys()).index(default_metric),
            format_func=lambda x: f"{x} — {metric_opts[x]['desc']}"
        )

    higher_is_better = metric_opts[chosen_metric]["higher_is_better"]

    # Recommendations Callout
    rec = generate_comparison_recommendations(
        evaluated_df,
        task=task,
        target_metric=chosen_metric,
        higher_is_better=higher_is_better
    )

    st.markdown(
        f"""
        <div class="callout-box">
            <h4 style="margin: 0 0 0.4rem 0; color: #166534;">🌟 Recommended Model: {rec['best_model']}</h4>
            <div style="font-size: 0.92rem; line-height: 1.5;">{rec['summary']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Bar chart ranking
    fig_bar = plot_model_comparison_bar(
        evaluated_df,
        metric_col=chosen_metric,
        metric_display_name=chosen_metric,
        higher_is_better=higher_is_better
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Overfitting Gap Analysis Chart
    st.markdown("#### 🔍 Overfitting & Generalization Gap Analysis")
    st.markdown("Compare performance on the Training split vs. the held-out Test split. Models with a large gap between training and testing show empirical signs of overfitting.")

    if task == "regression":
        fig_gap = plot_train_vs_test_gap(
            evaluated_df,
            test_metric_col="R² Score (Test)",
            train_metric_col="R² Score (Train)",
            metric_label="R² Score"
        )
    else:
        fig_gap = plot_train_vs_test_gap(
            evaluated_df,
            test_metric_col="Accuracy (Test)",
            train_metric_col="Accuracy (Train)",
            metric_label="Accuracy"
        )
    st.plotly_chart(fig_gap, use_container_width=True)

    st.markdown("---")

    # 5. Architectural Trade-offs & Insights
    st.markdown("### ⚖️ Architectural Trade-offs Across Models")
    for to in rec["trade_offs"]:
        st.markdown(to, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.page_link("pages/4_Make_Predictions.py", label="Test Live Predictions with the Best Model →", icon="🔮")


if __name__ == "__main__":
    main()
