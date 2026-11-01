"""
SmartML Model Registry & Compatibility Engine
Defines model architectures, hyperparameter mappings, compatibility validation,
and algorithm educational descriptions for both Regression and Classification tasks.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings
from scipy.stats import norm


class TabularHMMClassifier(BaseEstimator, ClassifierMixin):
    """
    Gaussian Hidden Markov / State-Emission Generative Classifier.
    Treats each class as a latent generative state with Gaussian emission distributions
    and estimated transition / prior probabilities.
    Fully compliant with the Scikit-learn estimator API (fit, predict, predict_proba).
    """

    def __init__(self, smoothing: float = 1e-4):
        self.smoothing = smoothing
        self.classes_: Optional[np.ndarray] = None
        self.class_priors_: Optional[np.ndarray] = None
        self.transition_matrix_: Optional[np.ndarray] = None
        self.means_: Optional[Dict[int, np.ndarray]] = None
        self.vars_: Optional[Dict[int, np.ndarray]] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        # Compute Class Priors / Stationary State distribution
        class_counts = np.bincount(y, minlength=n_classes)
        self.class_priors_ = (class_counts + 1.0) / (n_samples + n_classes)

        # Estimate Markov Transition Probabilities between sequential rows
        self.transition_matrix_ = np.ones((n_classes, n_classes)) * self.smoothing
        for t in range(len(y) - 1):
            s_from = y[t]
            s_to = y[t + 1]
            if s_from < n_classes and s_to < n_classes:
                self.transition_matrix_[s_from, s_to] += 1.0
        self.transition_matrix_ /= self.transition_matrix_.sum(axis=1, keepdims=True)

        # Estimate Gaussian Emission parameters (Mean and Variance) for each state
        self.means_ = {}
        self.vars_ = {}
        global_var = np.var(X, axis=0) + self.smoothing

        for c in self.classes_:
            X_c = X[y == c]
            if len(X_c) > 0:
                self.means_[c] = np.mean(X_c, axis=0)
                var_c = np.var(X_c, axis=0) + self.smoothing * global_var
                self.vars_[c] = var_c
            else:
                self.means_[c] = np.zeros(n_features)
                self.vars_[c] = global_var

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_probs = np.zeros((n_samples, n_classes))

        for idx, c in enumerate(self.classes_):
            mean = self.means_[c]
            var = self.vars_[c]
            # Gaussian log-emission density sum across features
            log_emission = -0.5 * np.sum(np.log(2.0 * np.pi * var) + ((X - mean) ** 2) / var, axis=1)
            # Combine emission log-likelihood with class prior
            log_prior = np.log(self.class_priors_[idx] + 1e-12)
            log_probs[:, idx] = log_emission + log_prior

        # Stable softmax normalization
        max_log = np.max(log_probs, axis=1, keepdims=True)
        exp_probs = np.exp(log_probs - max_log)
        probs = exp_probs / (np.sum(exp_probs, axis=1, keepdims=True) + 1e-12)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


def get_model_catalog(task: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns registered models for the selected task with educational metadata.
    """
    if task == "regression":
        return {
            "Linear Regression": {
                "name": "Linear Regression",
                "type": "Ordinary Least Squares",
                "description": "Calculates the line or hyperplane that minimizes the sum of squared differences between observed and predicted values.",
                "suitable_for": "Continuous numerical prediction with linear relationships and direct feature interpretability.",
                "limitations": "Sensitive to extreme outliers; assumes linear relationship; does not model complex interactions automatically.",
                "default_hyperparams": {"fit_intercept": True}
            },
            "Polynomial Regression": {
                "name": "Polynomial Regression",
                "type": "Non-linear Basis Expansion",
                "description": "Constructs polynomial and interaction features (degree 2) before fitting a linear model to capture curves and synergies.",
                "suitable_for": "Modeling non-linear trends, curves, acceleration, and synergistic feature interaction effects.",
                "limitations": "Risk of feature explosion with many columns; prone to severe overfitting on small datasets.",
                "default_hyperparams": {"degree": 2, "include_bias": False}
            },
            "Multiple Regression": {
                "name": "Multiple Regression",
                "type": "Multi-variable OLS",
                "description": "Extends simple linear regression to explicitly utilize multiple independent predictor features simultaneously.",
                "suitable_for": "Datasets with 2 or more distinct input features where each feature contributes partial predictive signal.",
                "limitations": "Requires at least 2 input features; sensitive to multicollinearity between predictors.",
                "default_hyperparams": {"fit_intercept": True}
            },
            "Ridge Regression": {
                "name": "Ridge Regression",
                "type": "L2 Regularized Linear Model",
                "description": "Applies an L2 penalty on coefficient magnitudes to shrink them toward zero, preventing variance inflation.",
                "suitable_for": "Datasets with correlated input features or high variance where ordinary least squares overfits.",
                "limitations": "Does not perform hard feature selection (coefficients shrink near zero but never reach absolute zero).",
                "default_hyperparams": {"alpha": 1.0}
            },
            "Random Forest Regressor": {
                "name": "Random Forest Regressor",
                "type": "Ensemble Bagging",
                "description": "Constructs a multitude of decision trees using bootstrap aggregation and random feature subsets, averaging predictions.",
                "suitable_for": "Complex non-linear relationships, tabular interactions, robust to outliers and mixed scale data.",
                "limitations": "Higher memory and computation overhead; cannot extrapolate beyond the training target range.",
                "default_hyperparams": {"n_estimators": 100, "max_depth": 10, "random_state": 42}
            }
        }
    else:  # classification
        return {
            "Logistic Regression": {
                "name": "Logistic Regression",
                "type": "Generalized Linear Model",
                "description": "Estimates the probability of discrete class outcomes using the log-odds (logit) formulation and sigmoid/softmax function.",
                "suitable_for": "Binary and multiclass classification baselines with linearly separable or log-linear boundaries.",
                "limitations": "Cannot capture complex non-linear boundaries without manual feature transformations.",
                "default_hyperparams": {"C": 1.0, "max_iter": 1000, "random_state": 42}
            },
            "k-Nearest Neighbors (k-NN)": {
                "name": "k-Nearest Neighbors (k-NN)",
                "type": "Instance-Based Non-Parametric",
                "description": "Classifies a query point based on the majority vote of the k closest training points in the scaled feature space.",
                "suitable_for": "Irregular non-linear decision boundaries; intuitive distance-based neighborhood analysis.",
                "limitations": "Sensitive to feature scales; computationally expensive at inference time with large datasets.",
                "default_hyperparams": {"n_neighbors": 5, "weights": "uniform"}
            },
            "Support Vector Machine (SVM)": {
                "name": "Support Vector Machine (SVM)",
                "type": "Kernel Margin Maximizer",
                "description": "Finds the optimal separating hyperplane that maximizes the geometric margin between classes using RBF kernel projection.",
                "suitable_for": "High-dimensional spaces, non-linear boundaries, cases where clear margins of separation exist.",
                "limitations": "Sensitive to feature scaling; slower training on very large datasets; calibrated probabilities require Platt scaling.",
                "default_hyperparams": {"C": 1.0, "kernel": "rbf", "probability": True, "random_state": 42}
            },
            "HMM": {
                "name": "HMM",
                "type": "Probabilistic State-Emission Generative Model",
                "description": "Models classes as latent states with Gaussian emissions and Markovian transition dependencies between adjacent records.",
                "suitable_for": "Sequential or ordered observations, generative modeling, probabilistic likelihood class ranking.",
                "limitations": "Assumes class-conditional Gaussian distribution and first-order Markov dependency.",
                "default_hyperparams": {"smoothing": 1e-3}
            },
            "Random Forest Classifier": {
                "name": "Random Forest Classifier",
                "type": "Ensemble Bagging",
                "description": "Averaged ensemble of decision trees voting on class membership with bootstrap sampling and feature randomization.",
                "suitable_for": "Tabular classification with complex non-linear boundaries, feature interactions, and resistant to overfitting.",
                "limitations": "Larger model size, slower inference than simple linear models.",
                "default_hyperparams": {"n_estimators": 100, "max_depth": 10, "random_state": 42}
            }
        }


def check_model_compatibility(
    model_name: str,
    task: str,
    num_features: int,
    num_samples: int,
    num_classes: int = 2
) -> Dict[str, Any]:
    """
    Validates model compatibility with dataset characteristics and provides detailed guidance.
    """
    is_compatible = True
    status = "Compatible"
    reasons = []
    suggestions = []

    # Regression specific checks
    if task == "regression":
        if model_name == "Multiple Regression":
            if num_features < 2:
                is_compatible = False
                status = "Incompatible"
                reasons.append(
                    f"Multiple Regression strictly requires at least 2 input features (current features: {num_features}). "
                    f"For a single input feature, please choose 'Linear Regression'."
                )
                suggestions.append("Select 'Linear Regression' or add more feature columns.")
            else:
                reasons.append(f"Multiple Regression is fully compatible with {num_features} input features.")

        elif model_name == "Polynomial Regression":
            if num_features > 15:
                status = "Warning"
                reasons.append(
                    f"Polynomial Regression with {num_features} features will generate over "
                    f"{int(num_features * (num_features + 1) / 2 + num_features)} terms, increasing risk of overfitting."
                )
                suggestions.append("Consider reducing feature count or using Ridge Regression with regularization.")
            elif num_samples < 25:
                status = "Warning"
                reasons.append(
                    f"Dataset has only {num_samples} samples. Polynomial degree 2 expansion may overfit quickly."
                )

        elif model_name == "Linear Regression":
            reasons.append("Linear Regression is compatible with your selected features.")

        elif model_name == "Ridge Regression":
            reasons.append("Ridge Regression applies L2 shrinkage and will benefit from standard feature scaling.")

        elif model_name == "Random Forest Regressor":
            reasons.append("Random Forest handles non-linear relationships without requiring feature scaling.")

    # Classification specific checks
    elif task == "classification":
        if num_classes < 2:
            is_compatible = False
            status = "Incompatible"
            reasons.append("Classification requires at least 2 distinct target classes.")
            suggestions.append("Check target column selection.")

        if model_name == "k-Nearest Neighbors (k-NN)":
            if num_samples < 6:
                is_compatible = False
                status = "Incompatible"
                reasons.append(f"Dataset has only {num_samples} samples, which is too small for k-NN (k=5).")
                suggestions.append("Use Logistic Regression or acquire more data.")
            else:
                reasons.append("k-NN requires normalized/standardized numerical features, which SmartML provides automatically.")

        elif model_name == "Support Vector Machine (SVM)":
            reasons.append("SVM with RBF kernel requires feature scaling and will compute calibrated probability estimates.")

        elif model_name == "Logistic Regression":
            reasons.append("Logistic Regression provides a solid probabilistic linear classification baseline.")

        elif model_name == "HMM":
            reasons.append("HMM models class-conditional Gaussian feature emissions and sequential transition patterns.")

        elif model_name == "Random Forest Classifier":
            reasons.append("Random Forest Classifier works effectively across tabular features without requiring strict scaling.")

    return {
        "is_compatible": is_compatible,
        "status": status,
        "reasons": reasons,
        "suggestions": suggestions
    }


def instantiate_model(model_name: str, task: str, custom_params: Optional[Dict[str, Any]] = None):
    """
    Instantiates the actual Scikit-learn or custom model object with user/default hyperparameters.
    """
    params = custom_params or {}

    if task == "regression":
        if model_name == "Linear Regression":
            return LinearRegression(fit_intercept=params.get("fit_intercept", True))
        elif model_name == "Multiple Regression":
            return LinearRegression(fit_intercept=params.get("fit_intercept", True))
        elif model_name == "Polynomial Regression":
            degree = int(params.get("degree", 2))
            return Pipeline([
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("linear", LinearRegression())
            ])
        elif model_name == "Ridge Regression":
            alpha = float(params.get("alpha", 1.0))
            return Ridge(alpha=alpha, random_state=42)
        elif model_name == "Random Forest Regressor":
            n_estimators = int(params.get("n_estimators", 100))
            max_depth = params.get("max_depth", 10)
            return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        else:
            raise ValueError(f"Unknown regression model: {model_name}")

    elif task == "classification":
        if model_name == "Logistic Regression":
            C = float(params.get("C", 1.0))
            return LogisticRegression(C=C, max_iter=1000, random_state=42)
        elif model_name == "k-Nearest Neighbors (k-NN)":
            n_neighbors = int(params.get("n_neighbors", 5))
            return KNeighborsClassifier(n_neighbors=n_neighbors, weights="uniform")
        elif model_name == "Support Vector Machine (SVM)":
            C = float(params.get("C", 1.0))
            kernel = params.get("kernel", "rbf")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                return SVC(C=C, kernel=kernel, probability=True, random_state=42)
        elif model_name == "HMM":
            smoothing = float(params.get("smoothing", 1e-3))
            return TabularHMMClassifier(smoothing=smoothing)
        elif model_name == "Random Forest Classifier":
            n_estimators = int(params.get("n_estimators", 100))
            max_depth = params.get("max_depth", 10)
            return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        else:
            raise ValueError(f"Unknown classification model: {model_name}")

    raise ValueError(f"Unknown task: {task}")
