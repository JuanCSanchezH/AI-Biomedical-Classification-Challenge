"""Simplified training pipeline for BR + XGBoost model."""

import logging
import os
import sys
from pathlib import Path

# Add src_clean to path
sys.path.append(str(Path(__file__).parent))

from data.loader import DataLoader
from features.vectorizer import FeaturePipeline, TextVectorizer

from models.br_xgboost_model import BRXGBoostClassifier

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train_br_xgboost_model():
    """Train the BR + XGBoost model."""
    logging.info("Starting BR + XGBoost model training")

    # Create output directory
    os.makedirs("models", exist_ok=True)

    # Load and prepare data
    logging.info("Loading and preparing data...")
    data_loader = DataLoader()
    data_info = data_loader.get_data_info()
    logging.info(f"Dataset info: {data_info}")

    X, y = data_loader.prepare_features(combine_title_abstract=True)
    X_train, X_test, y_train, y_test = data_loader.split_data(X, y, test_size=0.2)
    label_names = data_loader.get_label_names()

    logging.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
    logging.info(f"Label names: {label_names}")

    # Feature engineering
    logging.info("Performing feature engineering...")
    vectorizer = TextVectorizer(max_features=5000, ngram_range=(1, 2))
    feature_pipeline = FeaturePipeline(vectorizer)

    X_train_features = feature_pipeline.fit_transform(X_train)
    X_test_features = feature_pipeline.transform(X_test)

    logging.info(f"Feature engineering completed - Train: {X_train_features.shape}, Test: {X_test_features.shape}")

    # Save feature pipeline
    feature_pipeline.save_pipeline("models/feature_pipeline.pkl")

    # Train BR + XGBoost model
    logging.info("Training BR + XGBoost model...")
    model = BRXGBoostClassifier()
    model.fit(X_train_features, y_train, label_names=label_names)

    # Evaluate model
    metrics = model.evaluate(X_test_features, y_test)

    # Save model
    model.save("models/BR_xgboost_model.pkl")

    logging.info("Training completed successfully!")
    logging.info(f"Best weighted F1 score: {metrics['weighted_f1']:.4f}")

    return model, metrics


if __name__ == "__main__":
    train_br_xgboost_model()
