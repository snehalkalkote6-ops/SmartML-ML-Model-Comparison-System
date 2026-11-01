"""
SmartML Evaluation & Performance Metrics Engine
Computes real statistical machine learning metrics for Regression and Classification,
calculates confusion matrices, ROC curves, train-vs-test evaluation gaps,
and provides educational explanations for all metrics.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize


def evaluate_regression_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Computes regression evaluation metrics on test set and training set.
    """
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    mae_test = float(mean_absolute_error(y_test, y_pred_test))
    mse_test = float(mean_squared_error(y_test, y_pred_test))
    rmse_test = float(np.sqrt(mse_test))
    r2_test = float(r2_score(y_test, y_pred_test))

    mae_train = float(mean_absolute_error(y_train, y_pred_train))
    mse_train = float(mean_squared_error(y_train, y_pred_train))
    rmse_train = float(np.sqrt(mse_train))
    r2_train = float(r2_score(y_train, y_pred_train))

    residuals_test = y_test - y_pred_test

    return {
        "task": "regression",
        "test_metrics": {
            "mae": mae_test,
            "mse": mse_test,
            "rmse": rmse_test,
            "r2": r2_test
        },
        "train_metrics": {
            "mae": mae_train,
            "mse": mse_train,
            "rmse": rmse_train,
            "r2": r2_train
        },
        "predictions": {
            "y_test_actual": y_test.tolist(),
            "y_test_pred": y_pred_test.tolist(),
            "residuals": residuals_test.tolist()
        },
        "metric_explanations": {
            "MAE": "Mean Absolute Error: The average absolute difference between actual and predicted values. Expressed in the exact same units as the target. Lower is better.",
            "MSE": "Mean Squared Error: The average squared difference. Heavily penalizes larger errors due to squaring. Lower is better.",
            "RMSE": "Root Mean Squared Error: The square root of MSE. Like MAE, it is expressed in original target units, but gives higher weight to large outliers. Lower is better.",
            "R² Score": "Coefficient of Determination: Proportion of variance in the target explained by the model. 1.0 indicates perfect fit, 0.0 indicates baseline mean predictor, and negative values indicate performance worse than predicting the mean."
        }
    }


def evaluate_classification_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_labels: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """
    Computes classification evaluation metrics on test set and training set.
    Supports binary and multiclass metrics, confusion matrices, and ROC-AUC.
    """
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    classes_present = np.unique(np.concatenate([y_test, y_train]))
    n_classes = len(classes_present)
    labels_to_use = class_labels if class_labels is not None else [str(c) for c in classes_present]

    # Test Metrics
    acc_test = float(accuracy_score(y_test, y_pred_test))
    prec_macro = float(precision_score(y_test, y_pred_test, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_test, y_pred_test, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred_test, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred_test, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred_test, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred_test, average="weighted", zero_division=0))

    # Train Metrics (for overfitting check)
    acc_train = float(accuracy_score(y_train, y_pred_train))
    f1_train = float(f1_score(y_train, y_pred_train, average="weighted", zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test, labels=classes_present)
    cm_normalized = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    # Probabilities and ROC-AUC
    has_proba = hasattr(model, "predict_proba")
    roc_data = None
    roc_auc_val = None

    if has_proba:
        try:
            probs = model.predict_proba(X_test)
            if n_classes == 2:
                # Binary classification
                # Use probability of positive class (index 1)
                pos_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                fpr, tpr, thresholds = roc_curve(y_test, pos_probs)
                roc_auc_val = float(auc(fpr, tpr))
                roc_data = {
                    "type": "binary",
                    "curves": [
                        {
                            "name": f"ROC (AUC = {roc_auc_val:.3f})",
                            "fpr": fpr.tolist(),
                            "tpr": tpr.tolist()
                        }
                    ]
                }
            else:
                # Multiclass One-vs-Rest ROC
                y_test_bin = label_binarize(y_test, classes=classes_present)
                curves = []
                aucs = []
                for i, c in enumerate(classes_present):
                    if i < probs.shape[1]:
                        fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], probs[:, i])
                        auc_i = float(auc(fpr_i, tpr_i))
                        aucs.append(auc_i)
                        c_name = labels_to_use[i] if i < len(labels_to_use) else f"Class {c}"
                        curves.append({
                            "name": f"{c_name} (AUC = {auc_i:.3f})",
                            "fpr": fpr_i.tolist(),
                            "tpr": tpr_i.tolist()
                        })
                roc_auc_val = float(np.mean(aucs)) if aucs else None
                roc_data = {
                    "type": "multiclass",
                    "macro_auc": roc_auc_val,
                    "curves": curves
                }
        except Exception:
            roc_data = None
            roc_auc_val = None

    return {
        "task": "classification",
        "test_metrics": {
            "accuracy": acc_test,
            "precision_macro": prec_macro,
            "precision_weighted": prec_weighted,
            "recall_macro": rec_macro,
            "recall_weighted": rec_weighted,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "roc_auc": roc_auc_val
        },
        "train_metrics": {
            "accuracy": acc_train,
            "f1_weighted": f1_train
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
        "class_labels": [str(lbl) for lbl in labels_to_use],
        "has_probabilities": has_proba,
        "roc_data": roc_data,
        "metric_explanations": {
            "Accuracy": "Overall fraction of correct predictions across all classes. Best used when class distributions are balanced.",
            "Precision": "Proportion of positive identifications that were actually correct. High precision indicates few false positives.",
            "Recall": "Proportion of actual positives that were correctly identified. High recall indicates few false negatives (high sensitivity).",
            "F1-Score": "Harmonic mean of precision and recall. Provides a balanced metric especially when datasets exhibit class imbalance.",
            "ROC-AUC": "Area Under the Receiver Operating Characteristic Curve. Evaluates the classifier's ability to discriminate between classes across all decision thresholds. 1.0 is perfect; 0.5 is random chance."
        }
    }
