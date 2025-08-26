"""Data loading and preprocessing utilities."""

import logging
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")


class DataLoader:
    """Data loader for medical articles classification."""

    def __init__(self, data_path: str = "input/challenge_data.csv"):
        """Initialize the data loader.

        Args:
            data_path: Path to the CSV file containing the data
        """
        self.data_path = data_path
        self.data = None
        self.mlb = MultiLabelBinarizer()
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def load_data(self) -> pd.DataFrame:
        """Load the dataset from CSV file.

        Returns:
            DataFrame containing the loaded data
        """
        try:
            self.data = pd.read_csv(self.data_path, sep=";")
            logging.info(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def preprocess_text(self, text: str) -> str:
        """Preprocess text by cleaning and normalizing.

        Args:
            text: Input text to preprocess

        Returns:
            Preprocessed text
        """
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        MIN_TOKEN_LENGTH = 2
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > MIN_TOKEN_LENGTH
        ]

        return " ".join(tokens)

    def prepare_features(self, combine_title_abstract: bool = True) -> tuple:
        """Prepare features for modeling.

        Args:
            combine_title_abstract: Whether to combine title and abstract

        Returns:
            Tuple of (X, y) where X is features and y is labels
        """
        if self.data is None:
            self.load_data()

        # Preprocess text
        self.data["title_processed"] = self.data["title"].apply(self.preprocess_text)
        self.data["abstract_processed"] = self.data["abstract"].apply(self.preprocess_text)

        if combine_title_abstract:
            self.data["combined_text"] = self.data["title_processed"] + " " + self.data["abstract_processed"]
            X = self.data["combined_text"]
        else:
            X = self.data[["title_processed", "abstract_processed"]]

        # Prepare labels
        labels = [group.split("|") for group in self.data["group"]]
        y = self.mlb.fit_transform(labels)

        logging.info(f"Features prepared. X shape: {X.shape}, y shape: {y.shape}")
        logging.info(f"Label classes: {self.mlb.classes_}")

        return X, y

    def split_data(self, X, y, test_size: float = 0.2, random_state: int = 42) -> tuple:
        """Split data into train and test sets.

        Args:
            X: Features
            y: Labels
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )

        logging.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def get_label_names(self) -> list:
        """Get the names of the label classes.

        Returns:
            List of label class names
        """
        return list(self.mlb.classes_)

    def get_data_info(self) -> dict:
        """Get information about the dataset.

        Returns:
            Dictionary with dataset information
        """
        if self.data is None:
            self.load_data()

        info = {
            "total_samples": len(self.data),
            "unique_groups": self.data["group"].nunique(),
            "group_distribution": self.data["group"].value_counts().to_dict(),
            "title_length_stats": {
                "mean": self.data["title"].str.len().mean(),
                "median": self.data["title"].str.len().median(),
                "min": self.data["title"].str.len().min(),
                "max": self.data["title"].str.len().max(),
            },
            "abstract_length_stats": {
                "mean": self.data["abstract"].str.len().mean(),
                "median": self.data["abstract"].str.len().median(),
                "min": self.data["abstract"].str.len().min(),
                "max": self.data["abstract"].str.len().max(),
            },
        }

        return info
