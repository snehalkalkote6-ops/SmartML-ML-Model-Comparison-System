"""
SmartML Preprocessing Pipeline Engine
Implements strict leak-free preprocessing using Scikit-Learn:
- Train-test splitting before fitting transformations
- Numerical imputation (median) and standard scaling
- Categorical imputation (most frequent) and one-hot encoding
- Target label encoding with clean inverse mapping
- Transformed feature name tracking
- Single-record transformation for live inference
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


class SmartMLPipeline:
    """
    Manages the complete lifecycle of dataset splitting, feature transformation,
    and single-sample inference preprocessing.
    """

    def __init__(
        self,
        target_col: str,
        feature_cols: List[str],
        task: str,
        test_size: float = 0.25,
        random_state: int = 42,
        scale_numerical: bool = True
    ):
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.task = task
        self.test_size = test_size
        self.random_state = random_state
        self.scale_numerical = scale_numerical

        self.preprocessor: Optional[ColumnTransformer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.transformed_feature_names: List[str] = []
        self.is_fitted: bool = False
        self.target_classes: Optional[List[Any]] = None

    def fit_and_split(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Splits data into train and test sets, then fits preprocessing strictly
        on the training set and transforms both sets.
        Returns: (X_train_proc, X_test_proc, y_train, y_test, metadata)
        """
        # Drop rows where target is NaN to prevent target leakage
        clean_df = df.dropna(subset=[self.target_col]).copy()

        # Separate X and y
        X = clean_df[self.feature_cols].copy()
        y_raw = clean_df[self.target_col].copy()

        # Target encoding for classification if non-numeric or discrete
        if self.task == "classification":
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y_raw.astype(str))
            self.target_classes = list(self.label_encoder.classes_)
        else:
            y = y_raw.values.astype(float)
            self.label_encoder = None
            self.target_classes = None

        # Determine feature types within X
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = [col for col in self.feature_cols if col not in self.numeric_features]

        # Splitting strategy
        stratify_col = None
        stratified_used = False
        if self.task == "classification":
            # Check if all classes have at least 2 instances
            class_counts = pd.Series(y).value_counts()
            if class_counts.min() >= 2:
                stratify_col = y
                stratified_used = True

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_col
        )

        # Build ColumnTransformer transformers
        transformers = []

        if self.numeric_features:
            num_steps = [("imputer", SimpleImputer(strategy="median"))]
            if self.scale_numerical:
                num_steps.append(("scaler", StandardScaler()))
            num_pipe = Pipeline(steps=num_steps)
            transformers.append(("num", num_pipe, self.numeric_features))

        if self.categorical_features:
            cat_steps = [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]
            cat_pipe = Pipeline(steps=cat_steps)
            transformers.append(("cat", cat_pipe, self.categorical_features))

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )

        # Fit preprocessor strictly on training data
        X_train_proc = self.preprocessor.fit_transform(X_train_raw)
        X_test_proc = self.preprocessor.transform(X_test_raw)

        # Retrieve transformed feature names
        self.transformed_feature_names = self._extract_feature_names()
        self.is_fitted = True

        metadata = {
            "num_train_samples": len(X_train_raw),
            "num_test_samples": len(X_test_raw),
            "stratified_split": stratified_used,
            "numeric_feature_count": len(self.numeric_features),
            "categorical_feature_count": len(self.categorical_features),
            "transformed_feature_count": len(self.transformed_feature_names),
            "classes": self.target_classes
        }

        return X_train_proc, X_test_proc, y_train, y_test, metadata

    def transform_single(self, input_dict: Dict[str, Any]) -> np.ndarray:
        """
        Transforms a single user input record for inference.
        """
        if not self.is_fitted or self.preprocessor is None:
            raise ValueError("Preprocessing pipeline has not been fitted yet.")

        # Create 1-row DataFrame preserving exact feature columns
        df_single = pd.DataFrame([input_dict])[self.feature_cols]
        return self.preprocessor.transform(df_single)

    def decode_target(self, y_pred: Union[int, float, np.ndarray]) -> Any:
        """
        Converts encoded integer prediction back to original class label if classification.
        """
        if self.task == "classification" and self.label_encoder is not None:
            if isinstance(y_pred, (list, np.ndarray)):
                return self.label_encoder.inverse_transform(np.asarray(y_pred, dtype=int))
            return self.label_encoder.inverse_transform([int(y_pred)])[0]
        return y_pred

    def _extract_feature_names(self) -> List[str]:
        """
        Extracts clean transformed column names from ColumnTransformer.
        """
        feature_names = []
        if self.preprocessor is None:
            return feature_names

        for name, transformer, cols in self.preprocessor.transformers_:
            if name == "num":
                feature_names.extend(cols)
            elif name == "cat":
                encoder = transformer.named_steps["encoder"]
                cat_names = encoder.get_feature_names_out(cols)
                feature_names.extend(list(cat_names))

        return feature_names
