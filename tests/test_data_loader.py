"""Tests for data loader module."""

import numpy as np
import pandas as pd

from src.data.loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""

    def test_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert loader.data_path == "input/challenge_data.csv"
        assert loader.data is None
        assert loader.mlb is not None

    def test_load_data(self):
        """Test data loading functionality."""
        COLUMN_COUNT = 3
        loader = DataLoader()
        data = loader.load_data()

        assert isinstance(data, pd.DataFrame)
        assert data.shape[0] > 0
        assert data.shape[1] == COLUMN_COUNT
        assert list(data.columns) == ["title", "abstract", "group"]
        assert loader.data is not None

    def test_preprocess_text(self):
        """Test text preprocessing."""
        loader = DataLoader()

        # Test normal text
        text = "This is a test text with numbers 123 and special chars @#$"
        processed = loader.preprocess_text(text)
        assert isinstance(processed, str)
        assert "123" not in processed
        assert "@#$" not in processed
        assert processed.islower()

        # Test empty text
        processed_empty = loader.preprocess_text("")
        assert processed_empty == ""

        # Test None text
        processed_none = loader.preprocess_text(None)
        assert processed_none == ""

    def test_prepare_features(self):
        """Test feature preparation."""
        loader = DataLoader()
        loader.load_data()

        X, y = loader.prepare_features(combine_title_abstract=True)

        assert isinstance(X, pd.Series)
        assert isinstance(y, np.ndarray)
        assert len(X) == len(y)
        assert y.shape[1] > 0  # Should have multiple labels

    def test_split_data(self):
        """Test data splitting."""
        loader = DataLoader()
        loader.load_data()
        X, y = loader.prepare_features()

        X_train, X_test, y_train, y_test = loader.split_data(X, y, test_size=0.2)

        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def test_get_label_names(self):
        """Test getting label names."""
        loader = DataLoader()
        loader.load_data()
        loader.prepare_features()

        label_names = loader.get_label_names()
        assert isinstance(label_names, list)
        assert len(label_names) > 0
        assert all(isinstance(name, str) for name in label_names)

    def test_get_data_info(self):
        """Test getting data information."""
        loader = DataLoader()
        loader.load_data()

        info = loader.get_data_info()

        assert isinstance(info, dict)
        assert "total_samples" in info
        assert "unique_groups" in info
        assert "group_distribution" in info
        assert "title_length_stats" in info
        assert "abstract_length_stats" in info
        assert info["total_samples"] > 0
