"""
SmartML Intelligent Recommendations & Diagnostic Error Analysis Engine
Generates dynamic, evidence-grounded analytical summaries from actual evaluated metrics:
- Identifies top-performing algorithms based on user-chosen criteria
- Detects empirical evidence of overfitting or underfitting (Train vs. Test gap)
- Analyzes sample size reliability and class imbalance impact
- Delivers actionable next-step engineering suggestions
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def generate_single_model_recommendations(
    model_name: str,
    task: str,
    evaluation: Dict[str, Any],
    sample_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyzes single model training results and returns actionable diagnostic insights.
    """
    insights = []
    overfitting_alert = None
    next_steps = []

    test_metrics = evaluation.get("test_metrics", {})
    train_metrics = evaluation.get("train_metrics", {})
    num_test = sample_metadata.get("num_test_samples", 0)

    if task == "regression":
        r2_test = test_metrics.get("r2", 0.0)
        r2_train = train_metrics.get("r2", 0.0)
        rmse_test = test_metrics.get("rmse", 0.0)
        mae_test = test_metrics.get("mae", 0.0)

        # Performance assessment
        if r2_test > 0.85:
            insights.append(f"<b>Strong Predictive Power:</b> The model explains <b>{r2_test * 100:.1f}%</b> of the variance in the held-out test data (R² = {r2_test:.3f}).")
        elif r2_test > 0.60:
            insights.append(f"<b>Moderate Performance:</b> The model achieves an R² of <b>{r2_test:.3f}</b>. It captures general trends but exhibits noticeable residual spread.")
        elif r2_test >= 0.0:
            insights.append(f"<b>Weak Fit:</b> Low R² score of <b>{r2_test:.3f}</b>. The model explains little of the target variance beyond the sample mean.")
        else:
            insights.append(f"<b>Negative R² Alert ({r2_test:.3f}):</b> The model performs worse than a horizontal line predicting the target mean. This suggests non-linearity, unscaled features, or inappropriate assumptions.")

        # Overfitting check
        gap = r2_train - r2_test
        if r2_train > 0.85 and gap > 0.20:
            overfitting_alert = {
                "type": "Overfitting",
                "severity": "High",
                "message": f"Significant generalization gap detected: Training R² ({r2_train:.3f}) is much higher than Test R² ({r2_test:.3f}), gap = {gap:.3f}. The model memorized training patterns but struggles on unseen test samples."
            }
            next_steps.append("Apply regularization (e.g. Ridge Regression with higher alpha) or reduce model complexity (e.g., lower polynomial degree or tree depth).")
        elif gap > 0.10:
            overfitting_alert = {
                "type": "Mild Overfitting",
                "severity": "Medium",
                "message": f"Moderate gap between Training R² ({r2_train:.3f}) and Test R² ({r2_test:.3f})."
            }
            next_steps.append("Gather additional training records or perform feature selection to remove noisy predictors.")

        if r2_train < 0.40 and r2_test < 0.40:
            insights.append("<b>Potential Underfitting:</b> Both training and test scores are low. The model family may lack the capacity to express the underlying relationship, or key predictive features may be missing.")
            next_steps.append("Engineer interaction terms, try non-linear algorithms like Random Forest, or include more informative features.")

    else:  # classification
        acc_test = test_metrics.get("accuracy", 0.0)
        acc_train = train_metrics.get("accuracy", 0.0)
        f1_test = test_metrics.get("f1_weighted", 0.0)
        f1_train = train_metrics.get("f1_weighted", 0.0)

        # Performance assessment
        if acc_test >= 0.90:
            insights.append(f"<b>High Classification Accuracy:</b> Achieved <b>{acc_test * 100:.1f}%</b> accuracy and an F1-score of <b>{f1_test:.3f}</b> on held-out test data.")
        elif acc_test >= 0.75:
            insights.append(f"<b>Acceptable Performance:</b> Achieved <b>{acc_test * 100:.1f}%</b> accuracy (F1 = {f1_test:.3f}).")
        else:
            insights.append(f"<b>Lower Accuracy:</b> The model achieved <b>{acc_test * 100:.1f}%</b> accuracy. Misclassifications are present.")

        # Overfitting check
        gap = acc_train - acc_test
        if acc_train > 0.95 and gap > 0.15:
            overfitting_alert = {
                "type": "Overfitting",
                "severity": "High",
                "message": f"Training accuracy ({acc_train * 100:.1f}%) is substantially higher than test accuracy ({acc_test * 100:.1f}%). The decision boundary has adapted too tightly to training noise."
            }
            next_steps.append("Increase regularization (e.g. lower C in Logistic Regression or SVM, or limit max_depth in Random Forest).")

        # Class imbalance note
        if abs(acc_test - f1_test) > 0.10:
            insights.append("<b>Class Imbalance Notice:</b> There is a noticeable difference between overall accuracy and weighted F1-score, suggesting imbalanced class distributions.")
            next_steps.append("Evaluate precision and recall individually per class, or apply class weighting.")

    # Sample size caveat
    if num_test < 25:
        insights.append(f"<b>Statistical Variance Notice:</b> Test evaluation is based on {num_test} samples. On smaller test partitions, single errors have high percentage impact. K-fold cross-validation is recommended for formal verification.")

    if not next_steps:
        next_steps.append("Model is performing consistently. You can test new individual records on the 'Make Predictions' page.")
        next_steps.append("Compare with other candidate algorithms on the 'Model Comparison' page.")

    return {
        "model_name": model_name,
        "insights": insights,
        "overfitting_alert": overfitting_alert,
        "next_steps": next_steps
    }


def generate_comparison_recommendations(
    comparison_df: pd.DataFrame,
    task: str,
    target_metric: str,
    higher_is_better: bool = True
) -> Dict[str, Any]:
    """
    Ranks models, identifies the optimal algorithm, and summarizes architectural trade-offs.
    """
    if comparison_df.empty:
        return {
            "best_model": "None",
            "summary": "No models have been evaluated.",
            "trade_offs": []
        }

    df_sorted = comparison_df.sort_values(
        by=target_metric,
        ascending=not higher_is_better
    ).reset_index(drop=True)

    best_row = df_sorted.iloc[0]
    best_model = best_row["Model"]
    best_score = best_row[target_metric]

    if len(df_sorted) > 1:
        runner_up = df_sorted.iloc[1]["Model"]
        runner_score = df_sorted.iloc[1][target_metric]
        diff = abs(best_score - runner_score)
        summary = (
            f"Based on <b>{target_metric}</b> on the identical held-out test split, "
            f"<b>{best_model}</b> emerged as the top performer ({target_metric} = <b>{best_score:.4f}</b>), "
            f"outperforming runner-up <b>{runner_up}</b> ({target_metric} = {runner_score:.4f}) by a margin of {diff:.4f}."
        )
    else:
        summary = (
            f"<b>{best_model}</b> achieved a {target_metric} of <b>{best_score:.4f}</b> on the held-out test set."
        )

    trade_offs = []
    for _, row in df_sorted.iterrows():
        m = row["Model"]
        if "Linear" in m or "Ridge" in m:
            trade_offs.append(f"• <b>{m}:</b> Fast training, highly interpretable linear coefficients, but limited to linear boundaries.")
        elif "Polynomial" in m:
            trade_offs.append(f"• <b>{m}:</b> Captures non-linear curves via polynomial basis expansion, but risk of higher variance and feature explosion.")
        elif "Random Forest" in m:
            trade_offs.append(f"• <b>{m}:</b> Strong non-linear modeling and resistance to outliers, but requires more compute and acts as a black box.")
        elif "Logistic" in m:
            trade_offs.append(f"• <b>{m}:</b> Clean baseline with calibrated probabilistic odds, fast inference, but linear decision surface.")
        elif "k-NN" in m:
            trade_offs.append(f"• <b>{m}:</b> Intuitive local neighborhood voting, no assumptions about distribution, but sensitive to distance metric and slower inference.")
        elif "SVM" in m:
            trade_offs.append(f"• <b>{m}:</b> Powerful maximum-margin boundaries with RBF kernel, but computationally heavy and sensitive to scaling.")
        elif "HMM" in m:
            trade_offs.append(f"• <b>{m}:</b> Generative state-emission modeling capturing transition probabilities between consecutive observations.")

    return {
        "best_model": best_model,
        "best_score": best_score,
        "summary": summary,
        "ranked_models": df_sorted["Model"].tolist(),
        "trade_offs": trade_offs
    }
