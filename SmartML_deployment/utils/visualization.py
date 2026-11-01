"""
SmartML Plotly Interactive Visualization Engine
Builds modern, responsive, publication-quality Plotly charts for:
- Regression: Actual vs Predicted, Residual Plot, Error Distribution
- Classification: Interactive Confusion Matrix Heatmap, Multi-curve ROC Analysis, Class Probabilities
- Model Comparison: Metric Ranking Bars, Overfitting Gap (Train vs Test) comparisons
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# Modern corporate design palette
CHART_THEME = "plotly_white"
PRIMARY_COLOR = "#2563EB"   # Deep royal blue
SUCCESS_COLOR = "#10B981"   # Emerald green
ACCENT_COLOR = "#8B5CF6"    # Purple accent
WARN_COLOR = "#F59E0B"      # Amber
DANGER_COLOR = "#EF4444"    # Rose red


def plot_actual_vs_predicted(
    y_actual: List[float],
    y_pred: List[float],
    model_name: str = "Model"
) -> go.Figure:
    """
    Renders Actual vs. Predicted scatter plot with identity (y = x) line.
    """
    actual = np.array(y_actual)
    pred = np.array(y_pred)
    residuals = actual - pred

    min_val = min(actual.min(), pred.min())
    max_val = max(actual.max(), pred.max())
    padding = (max_val - min_val) * 0.05
    line_min = min_val - padding
    line_max = max_val + padding

    fig = go.Figure()

    # Identity reference line (y = x)
    fig.add_trace(go.Scatter(
        x=[line_min, line_max],
        y=[line_min, line_max],
        mode="lines",
        name="Perfect Fit (y = x)",
        line=dict(color="#94A3B8", dash="dash", width=2)
    ))

    # Prediction points
    fig.add_trace(go.Scatter(
        x=actual,
        y=pred,
        mode="markers",
        name="Predictions",
        marker=dict(
            color=PRIMARY_COLOR,
            size=9,
            opacity=0.8,
            line=dict(color="#1E3A8A", width=1)
        ),
        text=[f"Actual: {a:.2f}<br>Predicted: {p:.2f}<br>Residual: {r:.2f}"
              for a, p, r in zip(actual, pred, residuals)],
        hoverinfo="text"
    ))

    fig.update_layout(
        title=f"<b>Actual vs. Predicted Values</b> — {model_name}",
        xaxis_title="Actual Target Values (Ground Truth)",
        yaxis_title="Predicted Target Values",
        template=CHART_THEME,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


def plot_residuals(
    y_pred: List[float],
    residuals: List[float],
    model_name: str = "Model"
) -> go.Figure:
    """
    Renders Residuals vs. Predicted Values plot with zero error line.
    """
    pred = np.array(y_pred)
    res = np.array(residuals)

    fig = go.Figure()

    # Zero residual line
    fig.add_hline(
        y=0,
        line=dict(color="#EF4444", dash="dash", width=2),
        annotation_text="Zero Residual",
        annotation_position="bottom right"
    )

    # Residual scatter
    fig.add_trace(go.Scatter(
        x=pred,
        y=res,
        mode="markers",
        name="Residuals",
        marker=dict(
            color=res,
            colorscale="RdBu",
            reversescale=True,
            size=9,
            opacity=0.85,
            showscale=True,
            colorbar=dict(title="Residual", thickness=15)
        ),
        text=[f"Predicted: {p:.2f}<br>Residual: {r:.2f}" for p, r in zip(pred, res)],
        hoverinfo="text"
    ))

    fig.update_layout(
        title=f"<b>Residual Plot (Prediction Error vs Fitted Values)</b> — {model_name}",
        xaxis_title="Fitted / Predicted Values",
        yaxis_title="Residuals (Actual - Predicted)",
        template=CHART_THEME,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


def plot_error_distribution(
    residuals: List[float],
    model_name: str = "Model"
) -> go.Figure:
    """
    Renders a histogram of residuals with normality reference.
    """
    res = np.array(residuals)
    mean_res = np.mean(res)

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=res,
        name="Residuals",
        marker=dict(color=PRIMARY_COLOR, opacity=0.75, line=dict(color="#1E3A8A", width=1)),
        nbinsx=20
    ))

    fig.add_vline(
        x=0,
        line=dict(color="#EF4444", dash="dash", width=2),
        annotation_text="Zero Error",
        annotation_position="top left"
    )

    fig.add_vline(
        x=mean_res,
        line=dict(color="#10B981", dash="dot", width=2),
        annotation_text=f"Mean Error: {mean_res:.2f}",
        annotation_position="top right"
    )

    fig.update_layout(
        title=f"<b>Residual Error Distribution</b> — {model_name}",
        xaxis_title="Residual Value",
        yaxis_title="Frequency Count",
        template=CHART_THEME,
        bargap=0.08,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


def plot_confusion_matrix(
    cm: List[List[int]],
    class_labels: List[str],
    model_name: str = "Model"
) -> go.Figure:
    """
    Renders an interactive annotated Confusion Matrix Heatmap.
    """
    cm_arr = np.array(cm)
    total_samples = cm_arr.sum()
    text_matrix = []

    for i in range(len(class_labels)):
        row_text = []
        for j in range(len(class_labels)):
            count = cm_arr[i, j]
            pct = (count / total_samples * 100.0) if total_samples > 0 else 0
            row_text.append(f"<b>{count}</b><br><span style='font-size:11px'>({pct:.1f}%)</span>")
        text_matrix.append(row_text)

    fig = go.Figure(data=go.Heatmap(
        z=cm_arr,
        x=[f"Pred: {lbl}" for lbl in class_labels],
        y=[f"Actual: {lbl}" for lbl in class_labels],
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=14, color="white"),
        colorscale="Blues",
        showscale=True,
        colorbar=dict(title="Sample Count")
    ))

    fig.update_layout(
        title=f"<b>Confusion Matrix</b> — {model_name}",
        xaxis_title="Predicted Class Label",
        yaxis_title="Ground Truth Actual Label",
        template=CHART_THEME,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=60, r=40, t=60, b=50)
    )

    return fig


def plot_roc_curves(
    roc_data: Dict[str, Any],
    model_name: str = "Model"
) -> Optional[go.Figure]:
    """
    Renders binary or multiclass One-vs-Rest ROC curves with diagonal chance line.
    """
    if not roc_data or "curves" not in roc_data:
        return None

    fig = go.Figure()

    # Diagonal random guess line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Classifier (AUC = 0.50)",
        line=dict(color="#94A3B8", dash="dash", width=2)
    ))

    palette = [PRIMARY_COLOR, SUCCESS_COLOR, ACCENT_COLOR, WARN_COLOR, "#06B6D4", "#EC4899"]

    for idx, curve in enumerate(roc_data["curves"]):
        color = palette[idx % len(palette)]
        fig.add_trace(go.Scatter(
            x=curve["fpr"],
            y=curve["tpr"],
            mode="lines",
            name=curve["name"],
            line=dict(color=color, width=2.5)
        ))

    macro_auc_str = f" (Macro AUC: {roc_data['macro_auc']:.3f})" if "macro_auc" in roc_data and roc_data["macro_auc"] else ""

    fig.update_layout(
        title=f"<b>Receiver Operating Characteristic (ROC) Curve</b> — {model_name}{macro_auc_str}",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity / Recall)",
        xaxis=dict(range=[-0.02, 1.02]),
        yaxis=dict(range=[-0.02, 1.05]),
        template=CHART_THEME,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=70, b=40)
    )

    return fig


def plot_prediction_probabilities(
    probs: List[float],
    class_labels: List[str]
) -> go.Figure:
    """
    Renders horizontal probability bar chart for live classification inference.
    """
    pcts = [p * 100.0 for p in probs]
    df_prob = pd.DataFrame({
        "Class": [str(c) for c in class_labels],
        "Probability": pcts
    }).sort_values("Probability", ascending=True)

    fig = go.Figure(go.Bar(
        x=df_prob["Probability"],
        y=df_prob["Class"],
        orientation="h",
        marker=dict(
            color=df_prob["Probability"],
            colorscale="Viridis",
            line=dict(color="#1E3A8A", width=1)
        ),
        text=[f"{p:.1f}%" for p in df_prob["Probability"]],
        textposition="outside"
    ))

    fig.update_layout(
        title="<b>Class Prediction Confidence Probabilities</b>",
        xaxis_title="Confidence Percentage (%)",
        yaxis_title="Target Class",
        xaxis=dict(range=[0, 115]),
        template=CHART_THEME,
        height=260,
        margin=dict(l=40, r=40, t=50, b=30)
    )

    return fig


def plot_model_comparison_bar(
    comparison_df: pd.DataFrame,
    metric_col: str,
    metric_display_name: str,
    higher_is_better: bool = True
) -> go.Figure:
    """
    Renders comparative ranked bar chart across all trained models.
    """
    df_sorted = comparison_df.sort_values(
        by=metric_col,
        ascending=not higher_is_better
    ).copy()

    best_model = df_sorted.iloc[0]["Model"]
    colors = [
        SUCCESS_COLOR if m == best_model else PRIMARY_COLOR
        for m in df_sorted["Model"]
    ]

    fig = go.Figure(go.Bar(
        x=df_sorted["Model"],
        y=df_sorted[metric_col],
        marker=dict(color=colors, line=dict(color="#0F172A", width=1)),
        text=[f"{v:.4f}" for v in df_sorted[metric_col]],
        textposition="outside"
    ))

    fig.update_layout(
        title=f"<b>Model Performance Comparison: {metric_display_name}</b> (Best: {best_model})",
        xaxis_title="Machine Learning Model",
        yaxis_title=metric_display_name,
        template=CHART_THEME,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


def plot_train_vs_test_gap(
    comparison_df: pd.DataFrame,
    test_metric_col: str,
    train_metric_col: str,
    metric_label: str
) -> go.Figure:
    """
    Renders grouped bar chart showing Train vs. Test score gap to detect overfitting.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Train Metric",
        x=comparison_df["Model"],
        y=comparison_df[train_metric_col],
        marker_color="#93C5FD",
        text=[f"{v:.3f}" for v in comparison_df[train_metric_col]],
        textposition="outside"
    ))

    fig.add_trace(go.Bar(
        name="Test Metric (Held-out)",
        x=comparison_df["Model"],
        y=comparison_df[test_metric_col],
        marker_color=PRIMARY_COLOR,
        text=[f"{v:.3f}" for v in comparison_df[test_metric_col]],
        textposition="outside"
    ))

    fig.update_layout(
        title=f"<b>Generalization Gap Analysis: Train vs. Test {metric_label}</b>",
        xaxis_title="Model",
        yaxis_title=metric_label,
        barmode="group",
        template=CHART_THEME,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig
