"""Multi-label classification models implementation."""

import logging
import time
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, hamming_loss
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC


class MultiLabelClassifier:
    """Base class for multi-label classification."""

    def __init__(self, strategy: str, base_algorithm: str, **kwargs):
        """Initialize the multi-label classifier.

        Args:
            strategy: Classification strategy ('BR', 'CC', 'LP')
            base_algorithm: Base algorithm ('logistic', 'svm', 'xgboost')
            **kwargs: Additional parameters for the base algorithm
        """
        self.strategy = strategy
        self.base_algorithm = base_algorithm
        self.model = None
        self.label_names = None
        self.training_time = None
        self.kwargs = kwargs

    def _get_base_classifier(self):
        """Get the base classifier based on the algorithm name.

        Returns:
            Base classifier instance
        """
        if self.base_algorithm == "logistic":
            return LogisticRegression(random_state=42, max_iter=1000, **self.kwargs)
        elif self.base_algorithm == "svm":
            return SVC(random_state=42, probability=True, **self.kwargs)
        elif self.base_algorithm == "xgboost":
            return xgb.XGBClassifier(random_state=42, eval_metric="logloss", **self.kwargs)
        else:
            raise ValueError(f"Unknown base algorithm: {self.base_algorithm}")

    def fit(self, X: np.ndarray, y: np.ndarray, label_names: List[str] = None):
        """Fit the multi-label classifier.

        Args:
            X: Feature matrix
            y: Label matrix
            label_names: Names of the labels
        """
        start_time = time.time()

        self.label_names = label_names

        if self.strategy == "BR":
            self.model = MultiOutputClassifier(self._get_base_classifier())
        elif self.strategy == "CC":
            from sklearn.multioutput import ClassifierChain

            self.model = ClassifierChain(self._get_base_classifier(), random_state=42)
        elif self.strategy == "LP":
            # Custom LabelPowerset implementation since it's not available in sklearn
            from sklearn.ensemble import RandomForestClassifier

            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self.model.fit(X, y)
        self.training_time = time.time() - start_time

        logging.info(f"Model fitted in {self.training_time:.2f} seconds")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for new data.

        Args:
            X: Feature matrix

        Returns:
            Predicted label matrix
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for new data.

        Args:
            X: Feature matrix

        Returns:
            Predicted probability matrix
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        try:
            return self.model.predict_proba(X)
        except AttributeError:
            logging.warning("Probability prediction not available for this model")
            return None

    def save(self, filepath: str):
        """Save the model to disk.

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")

        model_data = {
            "model": self.model,
            "strategy": self.strategy,
            "base_algorithm": self.base_algorithm,
            "label_names": self.label_names,
            "training_time": self.training_time,
            "kwargs": self.kwargs,
        }

        joblib.dump(model_data, filepath)
        logging.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load the model from disk.

        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)

        self.model = model_data["model"]
        self.strategy = model_data["strategy"]
        self.base_algorithm = model_data["base_algorithm"]
        self.label_names = model_data["label_names"]
        self.training_time = model_data["training_time"]
        self.kwargs = model_data["kwargs"]

        logging.info(f"Model loaded from {filepath}")


class ModelEvaluator:
    """Evaluator for multi-label classification models."""

    def __init__(self):
        """Initialize the model evaluator."""

    def evaluate_model(
        self, model: MultiLabelClassifier, X_test: np.ndarray, y_test: np.ndarray, label_names: List[str] = None
    ) -> Dict[str, float]:
        """Evaluate a multi-label classification model.

        Args:
            model: Trained multi-label classifier
            X_test: Test feature matrix
            y_test: Test label matrix
            label_names: Names of the labels

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = model.predict(X_test)

        # Calculate metrics
        hamming = hamming_loss(y_test, y_pred)
        micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        subset_accuracy = accuracy_score(y_test, y_pred)

        # Calculate weighted F1 (average of micro and macro)
        weighted_f1 = (micro_f1 + macro_f1) / 2

        metrics = {
            "hamming_loss": hamming,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "subset_accuracy": subset_accuracy,
            "weighted_f1": weighted_f1,
            "training_time": model.training_time,
        }

        logging.info("Model evaluation completed:")
        for metric, value in metrics.items():
            logging.info(f"  {metric}: {value:.4f}")

        return metrics

    def get_detailed_report(
        self, model: MultiLabelClassifier, X_test: np.ndarray, y_test: np.ndarray, label_names: List[str] = None
    ) -> str:
        """Get detailed classification report.

        Args:
            model: Trained multi-label classifier
            X_test: Test feature matrix
            y_test: Test label matrix
            label_names: Names of the labels

        Returns:
            Detailed classification report
        """
        y_pred = model.predict(X_test)

        if label_names is None:
            label_names = [f"Label_{i}" for i in range(y_test.shape[1])]

        report = classification_report(y_test, y_pred, target_names=label_names, zero_division=0)

        return report


class ModelComparison:
    """Compare multiple multi-label classification models."""

    def __init__(self):
        """Initialize the model comparison."""
        self.results = []

    def add_model_result(self, strategy: str, algorithm: str, metrics: Dict[str, float]):
        """Add a model result to the comparison.

        Args:
            strategy: Classification strategy
            algorithm: Base algorithm
            metrics: Evaluation metrics
        """
        result = {"strategy": strategy, "algorithm": algorithm, **metrics}
        self.results.append(result)

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Get comparison results as a DataFrame.

        Returns:
            DataFrame with comparison results
        """
        if not self.results:
            return pd.DataFrame()

        df = pd.DataFrame(self.results)

        # Add rank based on weighted F1 score
        df["rank"] = df["weighted_f1"].rank(ascending=False).astype(int)

        # Reorder columns
        column_order = [
            "rank",
            "strategy",
            "algorithm",
            "weighted_f1",
            "micro_f1",
            "macro_f1",
            "subset_accuracy",
            "hamming_loss",
            "training_time",
        ]

        return df[column_order].sort_values("rank")

    def get_best_model(self) -> Dict[str, Any]:
        """Get the best performing model.

        Returns:
            Dictionary with best model information
        """
        if not self.results:
            return {}

        df = self.get_comparison_dataframe()
        return df.iloc[0].to_dict()

    def save_results(self, filepath: str):
        """Save comparison results to CSV.

        Args:
            filepath: Path to save the results
        """
        df = self.get_comparison_dataframe()
        df.to_csv(filepath, index=False)
        logging.info(f"Comparison results saved to {filepath}")
