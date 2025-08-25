"""Main pipeline for medical article classification challenge."""

import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))


from data.loader import DataLoader
from features.vectorizer import FeaturePipeline, TextVectorizer
from utils.visualization import ModelVisualizer, setup_plotting_style

from models.multilabel_models import ModelComparison, ModelEvaluator, MultiLabelClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("output/pipeline.log"), logging.StreamHandler()],
)


def setup_directories():
    """Create necessary directories."""
    directories = ["output", "models"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def train_and_evaluate_models():
    """Train and evaluate all 9 model combinations."""
    logging.info("Starting model training and evaluation pipeline")

    # Setup
    setup_directories()
    setup_plotting_style()

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

    # Define model combinations
    strategies = ["BR", "CC", "LP"]  # Binary Relevance, Classifier Chains, Label Powerset
    algorithms = ["logistic", "svm", "xgboost"]  # Logistic Regression, SVM, XGBoost

    # Initialize evaluator and comparison
    evaluator = ModelEvaluator()
    comparison = ModelComparison()
    visualizer = ModelVisualizer()

    # Train and evaluate all 9 combinations
    logging.info("Training and evaluating all 9 model combinations...")

    for strategy in strategies:
        for algorithm in algorithms:
            logging.info(f"Training {strategy} + {algorithm}...")

            try:
                # Create and train model
                model = MultiLabelClassifier(strategy=strategy, base_algorithm=algorithm)
                model.fit(X_train_features, y_train, label_names=label_names)

                # Evaluate model
                metrics = evaluator.evaluate_model(model, X_test_features, y_test, label_names)

                # Add to comparison
                comparison.add_model_result(strategy, algorithm, metrics)

                # Save model
                model.save(f"models/{strategy}_{algorithm}_model.pkl")

                logging.info(f"Completed {strategy} + {algorithm}")

            except Exception as e:
                logging.error(f"Error training {strategy} + {algorithm}: {e}")
                continue

    # Get comparison results
    comparison_df = comparison.get_comparison_dataframe()
    logging.info("Model comparison results:")
    logging.info(comparison_df.to_string())

    # Save comparison results
    comparison.save_results("output/model_comparison_results.csv")

    # Create visualizations
    logging.info("Creating visualizations...")

    # 1. Comparison charts for all 9 models
    visualizer.create_comparison_charts(comparison_df)

    # 2. Comparison table
    visualizer.create_comparison_table(comparison_df)

    # 3. Data exploration charts
    visualizer.create_data_exploration_charts(data_info)

    # Get best model
    best_model_info = comparison.get_best_model()
    logging.info(f"Best model: {best_model_info}")

    # 4. Best model chart
    visualizer.create_best_model_chart(
        best_model_info, f"{best_model_info['strategy']} + {best_model_info['algorithm'].title()}"
    )

    # 5. Confusion matrix for best model
    best_model = MultiLabelClassifier(strategy=best_model_info["strategy"], base_algorithm=best_model_info["algorithm"])
    best_model.load(f"models/{best_model_info['strategy']}_{best_model_info['algorithm']}_model.pkl")

    y_pred_best = best_model.predict(X_test_features)
    visualizer.create_confusion_matrix(y_test, y_pred_best, label_names)

    # Create output file with predictions
    logging.info("Creating output file with predictions...")
    create_output_file(data_loader, X_test, y_test, y_pred_best, label_names, best_model_info)

    logging.info("Pipeline completed successfully!")
    return best_model_info


def create_output_file(data_loader, X_test, y_test, y_pred, label_names, best_model_info):
    """Create output file with test data and predictions."""

    # Get original test data indices
    test_indices = data_loader.data.index[-len(X_test) :]  # Assuming test set is the last portion

    # Create output DataFrame
    output_data = data_loader.data.iloc[test_indices].copy()
    output_data["target"] = output_data["group"]  # Original labels

    # Convert predictions back to label format
    predicted_labels = []
    for pred in y_pred:
        labels = [label_names[i] for i, val in enumerate(pred) if val == 1]
        predicted_labels.append("|".join(labels) if labels else "none")

    output_data["predictions"] = predicted_labels

    # Add model information
    output_data["model_strategy"] = best_model_info["strategy"]
    output_data["model_algorithm"] = best_model_info["algorithm"]
    output_data["model_weighted_f1"] = best_model_info["weighted_f1"]

    # Save to CSV
    output_path = "output/test_predictions.csv"
    output_data.to_csv(output_path, index=False, sep=";")
    logging.info(f"Output file saved to {output_path}")

    # Print summary
    logging.info("Output file summary:")
    logging.info(f"  Total test samples: {len(output_data)}")
    logging.info(f"  Best model: {best_model_info['strategy']} + {best_model_info['algorithm']}")
    logging.info(f"  Weighted F1 Score: {best_model_info['weighted_f1']:.4f}")


def cleanup_unused_models(best_model_info):
    """Remove models that are not the best performing one."""
    logging.info("Cleaning up unused models...")

    strategies = ["BR", "CC", "LP"]
    algorithms = ["logistic", "svm", "xgboost"]

    for strategy in strategies:
        for algorithm in algorithms:
            model_file = f"models/{strategy}_{algorithm}_model.pkl"

            # Keep the best model and feature pipeline
            if strategy == best_model_info["strategy"] and algorithm == best_model_info["algorithm"]:
                logging.info(f"Keeping best model: {model_file}")
            elif os.path.exists(model_file):
                os.remove(model_file)
                logging.info(f"Removed: {model_file}")


if __name__ == "__main__":
    try:
        # Run the complete pipeline
        best_model_info = train_and_evaluate_models()

        # Cleanup unused models
        cleanup_unused_models(best_model_info)

        logging.info("Challenge completed successfully!")
        logging.info(f"Best model: {best_model_info['strategy']} + {best_model_info['algorithm']}")
        logging.info(f"Best weighted F1 score: {best_model_info['weighted_f1']:.4f}")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise
