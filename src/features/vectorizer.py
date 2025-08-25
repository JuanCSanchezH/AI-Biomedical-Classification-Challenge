"""Feature vectorization utilities for text data."""

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class TextVectorizer:
    """Text vectorization using TF-IDF."""

    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        """Initialize the text vectorizer.

        Args:
            max_features: Maximum number of features to extract
            ngram_range: Range of n-grams to extract
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.feature_names = None

    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        """Fit the vectorizer and transform the texts.

        Args:
            texts: Series of text documents

        Returns:
            TF-IDF matrix
        """
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features, ngram_range=self.ngram_range, min_df=2, max_df=0.95, stop_words="english"
        )

        X = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        logging.info(f"Vectorizer fitted. Features: {X.shape[1]}")
        return X

    def transform(self, texts: pd.Series) -> np.ndarray:
        """Transform texts using fitted vectorizer.

        Args:
            texts: Series of text documents

        Returns:
            TF-IDF matrix
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer must be fitted before transform")

        return self.vectorizer.transform(texts)

    def get_feature_names(self) -> list:
        """Get feature names.

        Returns:
            List of feature names
        """
        if self.feature_names is None:
            raise ValueError("Vectorizer must be fitted first")
        return list(self.feature_names)

    def save(self, filepath: str):
        """Save the vectorizer to disk.

        Args:
            filepath: Path to save the vectorizer
        """
        if self.vectorizer is None:
            raise ValueError("No vectorizer to save")

        joblib.dump(self.vectorizer, filepath)
        logging.info(f"Vectorizer saved to {filepath}")

    def load(self, filepath: str):
        """Load the vectorizer from disk.

        Args:
            filepath: Path to load the vectorizer from
        """
        self.vectorizer = joblib.load(filepath)
        self.feature_names = self.vectorizer.get_feature_names_out()
        logging.info(f"Vectorizer loaded from {filepath}")


class FeaturePipeline:
    """Pipeline for feature engineering."""

    def __init__(self, vectorizer: TextVectorizer):
        """Initialize the feature pipeline.

        Args:
            vectorizer: Text vectorizer instance
        """
        self.vectorizer = vectorizer
        self.pipeline = None

    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        """Fit and transform the texts.

        Args:
            texts: Series of text documents

        Returns:
            Feature matrix
        """
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: pd.Series) -> np.ndarray:
        """Transform texts using fitted pipeline.

        Args:
            texts: Series of text documents

        Returns:
            Feature matrix
        """
        return self.vectorizer.transform(texts)

    def save_pipeline(self, filepath: str):
        """Save the entire pipeline.

        Args:
            filepath: Path to save the pipeline
        """
        pipeline_data = {"vectorizer": self.vectorizer}
        joblib.dump(pipeline_data, filepath)
        logging.info(f"Pipeline saved to {filepath}")

    def load_pipeline(self, filepath: str):
        """Load the pipeline from disk.

        Args:
            filepath: Path to load the pipeline from
        """
        pipeline_data = joblib.load(filepath)
        self.vectorizer = pipeline_data["vectorizer"]
        logging.info(f"Pipeline loaded from {filepath}")
