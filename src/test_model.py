"""Script to test the trained BR + XGBoost model with new input."""

import sys
from pathlib import Path

import pandas as pd

# Add src_clean to path
sys.path.append(str(Path(__file__).parent))

from data.loader import DataLoader
from features.vectorizer import FeaturePipeline

from models.br_xgboost_model import BRXGBoostClassifier


class ModelTester:
    """Class to test the trained BR + XGBoost model with new input data."""

    def __init__(
        self,
        model_path: str = "models/BR_xgboost_model.pkl",
        feature_pipeline_path: str = "models/feature_pipeline.pkl",
    ):
        """Initialize the model tester.

        Args:
            model_path: Path to the trained model
            feature_pipeline_path: Path to the feature pipeline
        """
        self.model_path = model_path
        self.feature_pipeline_path = feature_pipeline_path
        self.model = None
        self.feature_pipeline = None
        self.data_loader = None
        self.label_names = None

    def load_model_and_pipeline(self):
        """Load the trained model and feature pipeline."""
        print("Loading trained BR + XGBoost model and feature pipeline...")

        # Load the model
        self.model = BRXGBoostClassifier()
        self.model.load(self.model_path)

        # Load the feature pipeline
        self.feature_pipeline = FeaturePipeline(None)
        self.feature_pipeline.load_pipeline(self.feature_pipeline_path)

        # Get label names from the model
        self.label_names = self.model.label_names

        print(f"Model loaded successfully. Label names: {self.label_names}")

    def preprocess_new_data(self, titles: list, abstracts: list) -> pd.Series:
        """Preprocess new input data.

        Args:
            titles: List of article titles
            abstracts: List of article abstracts

        Returns:
            Preprocessed text data
        """
        print("Preprocessing new input data...")

        # Create a temporary data loader for preprocessing
        self.data_loader = DataLoader()

        # Create a temporary DataFrame
        temp_data = pd.DataFrame({"title": titles, "abstract": abstracts})

        # Preprocess the text
        temp_data["title_processed"] = temp_data["title"].apply(self.data_loader.preprocess_text)
        temp_data["abstract_processed"] = temp_data["abstract"].apply(self.data_loader.preprocess_text)

        # Combine title and abstract
        temp_data["combined_text"] = temp_data["title_processed"] + " " + temp_data["abstract_processed"]

        print(f"Preprocessed {len(titles)} articles")
        return temp_data["combined_text"]

    def predict_single(self, title: str, abstract: str) -> dict:
        """Make prediction for a single article.

        Args:
            title: Article title
            abstract: Article abstract

        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            self.load_model_and_pipeline()

        # Preprocess the data
        processed_texts = self.preprocess_new_data([title], [abstract])

        # Transform to features
        print("Transforming text to features...")
        features = self.feature_pipeline.transform(processed_texts)

        # Make predictions
        print("Making predictions...")
        predictions = self.model.predict(features)

        # Convert predictions to label format
        predicted_labels = []
        for pred in predictions:
            labels = [self.label_names[i] for i, val in enumerate(pred) if val == 1]
            predicted_labels.append("|".join(labels) if labels else "none")

        # Create results
        result = {
            "title": title,
            "abstract": abstract,
            "predicted_labels": predicted_labels[0],
            "prediction_matrix": predictions[0],
            "label_names": self.label_names,
        }

        print(f"Prediction completed for: {title[:50]}...")
        return result

    def predict_multiple(self, titles: list, abstracts: list) -> list:
        """Make predictions on multiple articles.

        Args:
            titles: List of article titles
            abstracts: List of article abstracts

        Returns:
            List of prediction results
        """
        if self.model is None:
            self.load_model_and_pipeline()

        # Preprocess the data
        processed_texts = self.preprocess_new_data(titles, abstracts)

        # Transform to features
        print("Transforming text to features...")
        features = self.feature_pipeline.transform(processed_texts)

        # Make predictions
        print("Making predictions...")
        predictions = self.model.predict(features)

        # Convert predictions to label format
        predicted_labels = []
        for pred in predictions:
            labels = [self.label_names[i] for i, val in enumerate(pred) if val == 1]
            predicted_labels.append("|".join(labels) if labels else "none")

        # Create results
        results = []
        for _i, (title, abstract, pred_labels, pred_matrix) in enumerate(
            zip(titles, abstracts, predicted_labels, predictions, strict=False)
        ):
            results.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "predicted_labels": pred_labels,
                    "prediction_matrix": pred_matrix,
                    "label_names": self.label_names,
                }
            )

        print(f"Predictions completed for {len(titles)} articles")
        return results

    def save_predictions(self, results: list, output_path: str = "output/new_predictions.csv"):
        """Save predictions to CSV file.

        Args:
            results: List of prediction results from predict_multiple() method
            output_path: Path to save the results
        """
        # Create DataFrame
        df = pd.DataFrame(
            {
                "title": [r["title"] for r in results],
                "abstract": [r["abstract"] for r in results],
                "predicted_labels": [r["predicted_labels"] for r in results],
            }
        )

        # Add individual label predictions
        for i, label_name in enumerate(results[0]["label_names"]):
            df[f"predicted_{label_name}"] = [r["prediction_matrix"][i] for r in results]

        # Save to CSV
        df.to_csv(output_path, index=False, sep=";")
        print(f"Predictions saved to {output_path}")

        return df


def main():
    """Main function to demonstrate model testing."""

    # Initialize the tester
    tester = ModelTester()

    # Example 1: Test with a single article
    print("=" * 60)
    print("EXAMPLE 1: Testing with a single article")
    print("=" * 60)

    single_title = "Cardiac arrhythmia detection using machine learning algorithms"
    single_abstract = """
    This study presents a novel approach for detecting cardiac arrhythmias using
    machine learning techniques. We analyzed electrocardiogram data from 500 patients
    and developed a classification system that can identify various types of arrhythmias
    with high accuracy. The system achieved 95% sensitivity and 92% specificity in
    detecting atrial fibrillation and ventricular tachycardia.
    """

    result = tester.predict_single(single_title, single_abstract)

    print(f"Title: {result['title']}")
    print(f"Abstract: {result['abstract'][:100]}...")
    print(f"Predicted Labels: {result['predicted_labels']}")
    print("Individual Predictions:")
    for i, label in enumerate(result["label_names"]):
        prediction = "YES" if result["prediction_matrix"][i] == 1 else "NO"
        print(f"  - {label}: {prediction}")

    # Example 2: Test with multiple articles
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Testing with multiple articles")
    print("=" * 60)

    multiple_titles = [
        "Neural network analysis of brain imaging data for Alzheimer's detection",
        "Liver function tests in patients with cardiovascular disease",
        "Novel chemotherapy agents for lung cancer treatment",
        "Stroke rehabilitation using virtual reality technology",
    ]

    multiple_abstracts = [
        """
        This research investigates the use of neural networks for analyzing brain
        imaging data to detect early signs of Alzheimer's disease. We used MRI scans
        from 300 patients and achieved 88% accuracy in early detection.
        """,
        """
        We examined liver function in 150 patients with cardiovascular disease.
        Results showed significant correlations between cardiac markers and liver
        enzyme levels, suggesting potential organ interactions.
        """,
        """
        A new chemotherapy protocol was tested on 200 lung cancer patients.
        The treatment showed 40% improvement in survival rates compared to
        standard chemotherapy regimens.
        """,
        """
        Virtual reality technology was implemented for stroke rehabilitation
        in 80 patients. The system showed 60% improvement in motor function
        recovery compared to traditional rehabilitation methods.
        """,
    ]

    results = tester.predict_multiple(multiple_titles, multiple_abstracts)

    for i, result in enumerate(results):
        print(f"\nArticle {i+1}:")
        print(f"  Title: {result['title']}")
        print(f"  Predicted Labels: {result['predicted_labels']}")

    # Save predictions
    tester.save_predictions(results, "output/multiple_predictions.csv")

    print("\n" + "=" * 60)
    print("Testing completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
