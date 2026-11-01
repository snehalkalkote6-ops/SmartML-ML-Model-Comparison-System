"""
SmartML Test Suite
Automated unit and integration testing across data loaders, preprocessor,
all regression/classification models, evaluation metrics, and inference.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_loader import (
    load_sample_dataset,
    get_dataset_summary,
    detect_prediction_task,
    validate_dataset_for_task
)
from utils.preprocessing import SmartMLPipeline
from utils.model_registry import (
    get_model_catalog,
    check_model_compatibility,
    instantiate_model,
    TabularHMMClassifier
)
from utils.evaluation import (
    evaluate_regression_model,
    evaluate_classification_model
)
from utils.recommendations import (
    generate_single_model_recommendations,
    generate_comparison_recommendations
)


class TestSmartMLPlatform(unittest.TestCase):

    def test_sample_datasets_load(self):
        """Verifies sample regression and classification datasets load properly."""
        df_reg, target_reg, task_reg, desc_reg = load_sample_dataset("regression")
        self.assertIsNotNone(df_reg)
        self.assertGreater(len(df_reg), 20)
        self.assertIn(target_reg, df_reg.columns)
        self.assertEqual(task_reg, "regression")

        df_clf, target_clf, task_clf, desc_clf = load_sample_dataset("classification")
        self.assertIsNotNone(df_clf)
        self.assertGreater(len(df_clf), 20)
        self.assertIn(target_clf, df_clf.columns)
        self.assertEqual(task_clf, "classification")

    def test_data_profiling_and_task_detection(self):
        """Verifies data summary metrics and intelligent task detection."""
        df_reg, target_reg, _, _ = load_sample_dataset("regression")
        summary = get_dataset_summary(df_reg)
        self.assertEqual(summary["num_rows"], len(df_reg))
        self.assertEqual(summary["num_cols"], len(df_reg.columns))
        self.assertIn("columns_info", summary)

        task_detected, conf, expl = detect_prediction_task(df_reg, target_reg)
        self.assertEqual(task_detected, "regression")

        df_clf, target_clf, _, _ = load_sample_dataset("classification")
        task_detected_clf, _, _ = detect_prediction_task(df_clf, target_clf)
        self.assertEqual(task_detected_clf, "classification")

    def test_preprocessing_pipeline_leak_free(self):
        """Ensures leak-free preprocessing splits before fitting and transforms single record."""
        df_reg, target_reg, _, _ = load_sample_dataset("regression")
        features = [c for c in df_reg.columns if c != target_reg]

        pipeline = SmartMLPipeline(
            target_col=target_reg,
            feature_cols=features,
            task="regression",
            test_size=0.25,
            random_state=42
        )
        X_train_proc, X_test_proc, y_train, y_test, meta = pipeline.fit_and_split(df_reg)

        self.assertEqual(len(X_train_proc), meta["num_train_samples"])
        self.assertEqual(len(X_test_proc), meta["num_test_samples"])
        self.assertGreater(meta["transformed_feature_count"], 0)

        # Single record transformation
        single_input = {col: df_reg[col].iloc[0] for col in features}
        X_single = pipeline.transform_single(single_input)
        self.assertEqual(X_single.shape[1], X_train_proc.shape[1])

    def test_all_regression_models(self):
        """Trains and evaluates all supported regression algorithms."""
        df_reg, target_reg, _, _ = load_sample_dataset("regression")
        features = [c for c in df_reg.columns if c != target_reg]

        pipeline = SmartMLPipeline(
            target_col=target_reg,
            feature_cols=features,
            task="regression",
            test_size=0.25,
            random_state=42
        )
        X_train_proc, X_test_proc, y_train, y_test, _ = pipeline.fit_and_split(df_reg)

        reg_models = [
            "Linear Regression",
            "Polynomial Regression",
            "Multiple Regression",
            "Ridge Regression",
            "Random Forest Regressor"
        ]

        for m_name in reg_models:
            compat = check_model_compatibility(m_name, "regression", len(features), len(df_reg))
            self.assertTrue(compat["is_compatible"], f"{m_name} should be compatible")

            model = instantiate_model(m_name, "regression")
            model.fit(X_train_proc, y_train)
            results = evaluate_regression_model(model, X_train_proc, y_train, X_test_proc, y_test)

            self.assertIn("r2", results["test_metrics"])
            self.assertIn("rmse", results["test_metrics"])
            self.assertIn("mae", results["test_metrics"])
            self.assertGreater(len(results["predictions"]["residuals"]), 0)

    def test_all_classification_models(self):
        """Trains and evaluates all supported classification algorithms including HMM."""
        df_clf, target_clf, _, _ = load_sample_dataset("classification")
        features = [c for c in df_clf.columns if c != target_clf]

        pipeline = SmartMLPipeline(
            target_col=target_clf,
            feature_cols=features,
            task="classification",
            test_size=0.25,
            random_state=42
        )
        X_train_proc, X_test_proc, y_train, y_test, meta = pipeline.fit_and_split(df_clf)

        clf_models = [
            "Logistic Regression",
            "k-Nearest Neighbors (k-NN)",
            "Support Vector Machine (SVM)",
            "HMM",
            "Random Forest Classifier"
        ]

        for m_name in clf_models:
            compat = check_model_compatibility(m_name, "classification", len(features), len(df_clf), num_classes=2)
            self.assertTrue(compat["is_compatible"], f"{m_name} should be compatible")

            model = instantiate_model(m_name, "classification")
            model.fit(X_train_proc, y_train)
            results = evaluate_classification_model(
                model,
                X_train_proc,
                y_train,
                X_test_proc,
                y_test,
                class_labels=meta["classes"]
            )

            self.assertIn("accuracy", results["test_metrics"])
            self.assertIn("f1_weighted", results["test_metrics"])
            self.assertIsNotNone(results["confusion_matrix"])

            # Test single record prediction and decode
            single_record = {col: df_clf[col].iloc[0] for col in features}
            X_single = pipeline.transform_single(single_record)
            pred = model.predict(X_single)
            decoded = pipeline.decode_target(pred[0])
            self.assertIn(str(decoded), ["Approved", "Rejected"])

    def test_recommendation_and_tradeoffs(self):
        """Verifies diagnostic recommendations and multi-model ranking logic."""
        mock_eval = {
            "test_metrics": {"r2": 0.88, "rmse": 3.2, "mae": 2.5},
            "train_metrics": {"r2": 0.90, "rmse": 3.0, "mae": 2.3}
        }
        meta = {"num_test_samples": 25}
        diag = generate_single_model_recommendations("Linear Regression", "regression", mock_eval, meta)
        self.assertGreater(len(diag["insights"]), 0)

        # Multi-model ranking
        comp_df = pd.DataFrame([
            {"Model": "Linear Regression", "RMSE (Test)": 3.5, "is_evaluated": True},
            {"Model": "Random Forest Regressor", "RMSE (Test)": 2.8, "is_evaluated": True}
        ])
        comp_rec = generate_comparison_recommendations(comp_df, "regression", "RMSE (Test)", higher_is_better=False)
        self.assertEqual(comp_rec["best_model"], "Random Forest Regressor")


if __name__ == "__main__":
    unittest.main()
