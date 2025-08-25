"""
Models module for multi-label classification of medical articles.
Implements three strategies: Binary Relevance, Classifier Chains, Label Powerset
with three base algorithms: Logistic Regression, XGBoost, SVM.
"""

import warnings

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset

warnings.filterwarnings("ignore")


class MultiLabelClassifier:
    """Base class for multi-label classifiers."""

    def __init__(self, strategy, algorithm, **kwargs):
        """
        Initialize MultiLabelClassifier.

        Args:
            strategy (str): Classification strategy ('BR', 'CC', 'LP')
            algorithm (str): Base algorithm ('LR', 'XGB', 'SVM')
            **kwargs: Additional parameters for the algorithm
        """
        self.strategy = strategy
        self.algorithm = algorithm
        self.model = None
        self.kwargs = kwargs
        self._create_model()

    def _create_model(self):
        """Create the multi-label classification model."""
        base_classifier = self._get_base_classifier()

        if self.strategy == "BR":
            self.model = BinaryRelevance(base_classifier)
        elif self.strategy == "CC":
            self.model = ClassifierChain(base_classifier)
        elif self.strategy == "LP":
            self.model = LabelPowerset(base_classifier)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _get_base_classifier(self):
        """Get the base classifier based on the algorithm."""
        if self.algorithm == "LR":
            return LogisticRegression(random_state=42, **self.kwargs)
        elif self.algorithm == "XGB":
            return xgb.XGBClassifier(random_state=42, **self.kwargs)
        elif self.algorithm == "SVM":
            return SVC(random_state=42, **self.kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def fit(self, X, y):
        """
        Fit the model to the training data.

        Args:
            X: Training features
            y: Training labels
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict labels for new data.

        Args:
            X: Test features

        Returns:
            array: Predicted labels
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict probabilities for new data.

        Args:
            X: Test features

        Returns:
            array: Predicted probabilities
        """
        return self.model.predict_proba(X)

    def get_model_name(self):
        """Get the model name for identification."""
        return f"{self.strategy}_{self.algorithm}"


class ModelFactory:
    """Factory class to create different model combinations."""

    @staticmethod
    def create_all_models():
        """
        Create all 9 model combinations (3 strategies × 3 algorithms).

        Returns:
            dict: Dictionary of models with keys as model names
        """
        strategies = ["BR", "CC", "LP"]
        algorithms = ["LR", "XGB", "SVM"]

        models = {}

        for strategy in strategies:
            for algorithm in algorithms:
                model_name = f"{strategy}_{algorithm}"

                # Adjust parameters based on algorithm
                if algorithm == "SVM":
                    # Use smaller dataset for SVM due to computational constraints
                    kwargs = {"C": 0.1, "gamma": "scale"}
                elif algorithm == "XGB":
                    kwargs = {"n_estimators": 50, "max_depth": 4}
                else:  # LR
                    kwargs = {"C": 1.0, "max_iter": 1000}

                models[model_name] = MultiLabelClassifier(strategy, algorithm, **kwargs)

        return models

    @staticmethod
    def create_specific_model(strategy, algorithm, **kwargs):
        """
        Create a specific model combination.

        Args:
            strategy (str): Classification strategy
            algorithm (str): Base algorithm
            **kwargs: Additional parameters

        Returns:
            MultiLabelClassifier: The created model
        """
        return MultiLabelClassifier(strategy, algorithm, **kwargs)


class ModelTrainer:
    """Class to handle model training and evaluation."""

    def __init__(self):
        """Initialize ModelTrainer."""
        self.models = {}
        self.results = {}

    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train all 9 model combinations.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels

        Returns:
            dict: Results for all models
        """
        print("Creating all model combinations...")
        self.models = ModelFactory.create_all_models()

        results = {}

        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")

            try:
                # Train the model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Store results
                results[model_name] = {"model": model, "predictions": y_pred, "true_labels": y_test}

                print(f"✓ {model_name} trained successfully")

            except Exception as e:
                print(f"✗ Error training {model_name}: {e}")
                results[model_name] = {"model": None, "predictions": None, "true_labels": y_test, "error": str(e)}

        self.results = results
        return results

    def get_best_model(self, metric="f1_micro"):
        """
        Get the best model based on a specific metric.

        Args:
            metric (str): Metric to use for comparison

        Returns:
            tuple: (best_model_name, best_model)
        """
        if not self.results:
            raise ValueError("No results available. Train models first.")

        # This will be used after evaluation
        # For now, return the first successful model
        for model_name, result in self.results.items():
            if result["model"] is not None:
                return model_name, result["model"]

        return None, None
