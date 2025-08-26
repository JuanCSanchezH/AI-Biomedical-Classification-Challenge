"""Binary Relevance + XGBoost model implementation."""

import logging
import time
from typing import List

import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.multioutput import MultiOutputClassifier


class BRXGBoostClassifier:
    """Binary Relevance + XGBoost classifier for multi-label classification."""

    def __init__(self, **kwargs):
        """Initialize the BR + XGBoost classifier.

        Args:
            **kwargs: Additional parameters for XGBoost
        """
        self.model = None
        self.label_names = None
        self.training_time = None
        self.kwargs = kwargs

    def _get_xgboost_classifier(self):
        """Get the XGBoost classifier.

        Returns:
            XGBoost classifier instance
        """
        return xgb.XGBClassifier(random_state=42, eval_metric="logloss", **self.kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, label_names: List[str] = None):
        """Fit the BR + XGBoost classifier.

        Args:
            X: Feature matrix
            y: Label matrix
            label_names: Names of the labels
        """
        start_time = time.time()

        self.label_names = label_names
        self.model = MultiOutputClassifier(self._get_xgboost_classifier())
        self.model.fit(X, y)
        self.training_time = time.time() - start_time

        logging.info(f"BR + XGBoost model fitted in {self.training_time:.2f} seconds")

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

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model.

        Args:
            X_test: Test feature matrix
            y_test: Test label matrix

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test)

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
            "training_time": self.training_time,
        }

        logging.info("Model evaluation completed:")
        for metric, value in metrics.items():
            logging.info(f"  {metric}: {value:.4f}")

        return metrics

    def save(self, filepath: str):
        """Save the model to disk.

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")

        model_data = {
            "model": self.model,
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
        self.label_names = model_data["label_names"]
        self.training_time = model_data["training_time"]
        self.kwargs = model_data["kwargs"]

        logging.info(f"Model loaded from {filepath}")
