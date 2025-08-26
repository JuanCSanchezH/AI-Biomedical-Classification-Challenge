"""Tests for BR + XGBoost model."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src_clean to path
sys.path.append(str(Path(__file__).parent.parent / "src_clean"))

from src.models.br_xgboost_model import BRXGBoostClassifier


class TestBRXGBoostModel:
    """Test cases for BRXGBoostClassifier class."""

    def test_init(self):
        """Test BRXGBoostClassifier initialization."""
        model = BRXGBoostClassifier()
        assert model.model is None
        assert model.label_names is None
        assert model.training_time is None

    def test_get_xgboost_classifier(self):
        """Test XGBoost classifier creation."""
        model = BRXGBoostClassifier()
        classifier = model._get_xgboost_classifier()
        assert classifier is not None
        assert hasattr(classifier, "fit")
        assert hasattr(classifier, "predict")

    def test_save_and_load(self, tmp_path):
        """Test model saving and loading."""
        model = BRXGBoostClassifier()

        # Create dummy data
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, (10, 3))
        label_names = ["label1", "label2", "label3"]

        # Fit model
        model.fit(X, y, label_names)

        # Save model
        model_path = tmp_path / "test_model.pkl"
        model.save(str(model_path))

        # Load model
        new_model = BRXGBoostClassifier()
        new_model.load(str(model_path))

        # Check if loaded correctly
        assert new_model.label_names == label_names
        assert new_model.training_time == model.training_time
        assert new_model.model is not None

    def test_predict_before_fit(self):
        """Test that prediction fails before fitting."""
        model = BRXGBoostClassifier()
        X = np.random.rand(5, 3)

        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            model.predict(X)

    def test_evaluate(self):
        """Test model evaluation."""
        model = BRXGBoostClassifier()

        # Create dummy data
        X_train = np.random.rand(20, 5)
        y_train = np.random.randint(0, 2, (20, 3))
        X_test = np.random.rand(10, 5)
        y_test = np.random.randint(0, 2, (10, 3))
        label_names = ["label1", "label2", "label3"]

        # Fit model
        model.fit(X_train, y_train, label_names)

        # Evaluate
        metrics = model.evaluate(X_test, y_test)

        # Check metrics
        assert "hamming_loss" in metrics
        assert "micro_f1" in metrics
        assert "macro_f1" in metrics
        assert "subset_accuracy" in metrics
        assert "weighted_f1" in metrics
        assert "training_time" in metrics

        # Check metric ranges
        assert 0 <= metrics["hamming_loss"] <= 1
        assert 0 <= metrics["micro_f1"] <= 1
        assert 0 <= metrics["macro_f1"] <= 1
        assert 0 <= metrics["subset_accuracy"] <= 1
        assert 0 <= metrics["weighted_f1"] <= 1
        assert metrics["training_time"] >= 0
