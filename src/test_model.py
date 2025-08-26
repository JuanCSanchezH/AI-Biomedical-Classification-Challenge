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
"""Enhanced script to test the trained BR + XGBoost model with CSV input and comprehensive metrics."""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    hamming_loss, f1_score, accuracy_score, 
    classification_report, confusion_matrix
)

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data.loader import DataLoader
from features.vectorizer import FeaturePipeline
from models.br_xgboost_model import BRXGBoostClassifier


class EnhancedModelTester:
    """Enhanced class to test the trained BR + XGBoost model with CSV input and metrics."""

    def __init__(self, model_path: str = "models/BR_xgboost_model.pkl",
                 feature_pipeline_path: str = "models/feature_pipeline.pkl"):
        """Initialize the enhanced model tester.

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
        temp_data = pd.DataFrame({
            "title": titles,
            "abstract": abstracts
        })

        # Preprocess the text
        temp_data["title_processed"] = temp_data["title"].apply(self.data_loader.preprocess_text)
        temp_data["abstract_processed"] = temp_data["abstract"].apply(self.data_loader.preprocess_text)

        # Combine title and abstract
        temp_data["combined_text"] = (
            temp_data["title_processed"] + " " + temp_data["abstract_processed"]
        )

        print(f"Preprocessed {len(titles)} articles")
        return temp_data["combined_text"]

    def predict_csv(self, csv_path: str, output_path: str = None) -> dict:
        """Predict labels for articles in a CSV file.

        Args:
            csv_path: Path to the CSV file with columns: title, abstract, group
            output_path: Path to save the results (optional)

        Returns:
            Dictionary with predictions and metrics
        """
        if self.model is None:
            self.load_model_and_pipeline()

        # Load CSV file
        print(f"Loading CSV file: {csv_path}")
        try:
            df = pd.read_csv(csv_path, sep=';')
        except:
            # Try comma separator
            df = pd.read_csv(csv_path, sep=',')
        
        # Validate required columns
        required_columns = ['title', 'abstract', 'group']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"Loaded {len(df)} articles from CSV")

        # Preprocess the data
        processed_texts = self.preprocess_new_data(df['title'].tolist(), df['abstract'].tolist())

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
            predicted_labels.append('|'.join(labels) if labels else 'none')

        # Add predictions to DataFrame
        df['group_predicted'] = predicted_labels

        # Prepare true labels for evaluation (if group column exists)
        if 'group' in df.columns and not df['group'].isna().all():
            # Convert true labels to binary format for evaluation
            true_labels = self._convert_labels_to_binary(df['group'].tolist())
            pred_labels_binary = predictions
            
            # Calculate metrics
            metrics = self._calculate_metrics(true_labels, pred_labels_binary)
            
            # Create confusion matrix
            self._create_confusion_matrix(true_labels, pred_labels_binary, output_path)
        else:
            metrics = None
            print("No true labels found for evaluation")

        # Save results
        if output_path:
            df.to_csv(output_path, index=False, sep=';')
            print(f"Results saved to: {output_path}")

        return {
            'dataframe': df,
            'predictions': predictions,
            'predicted_labels': predicted_labels,
            'metrics': metrics,
            'label_names': self.label_names
        }

    def _convert_labels_to_binary(self, labels: list) -> np.ndarray:
        """Convert label strings to binary format for evaluation.

        Args:
            labels: List of label strings

        Returns:
            Binary label matrix
        """
        binary_labels = []
        for label in labels:
            if pd.isna(label):
                binary_labels.append([0, 0, 0, 0])
            else:
                label_list = label.split('|')
                binary_row = [1 if label_name in label_list else 0 for label_name in self.label_names]
                binary_labels.append(binary_row)
        
        return np.array(binary_labels)

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate comprehensive evaluation metrics.

        Args:
            y_true: True labels in binary format
            y_pred: Predicted labels in binary format

        Returns:
            Dictionary with evaluation metrics
        """
        print("Calculating evaluation metrics...")
        
        # Calculate metrics
        hamming = hamming_loss(y_true, y_pred)
        micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        subset_accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate weighted F1 (main metric)
        weighted_f1 = (micro_f1 + macro_f1) / 2
        
        # Calculate per-label metrics
        per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        per_label_metrics = dict(zip(self.label_names, per_label_f1))
        
        metrics = {
            'hamming_loss': hamming,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'subset_accuracy': subset_accuracy,
            'weighted_f1': weighted_f1,
            'per_label_f1': per_label_metrics
        }
        
        # Print metrics
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        print(f"Weighted F1 Score (Main Metric): {weighted_f1:.4f}")
        print(f"Micro F1 Score: {micro_f1:.4f}")
        print(f"Macro F1 Score: {macro_f1:.4f}")
        print(f"Subset Accuracy: {subset_accuracy:.4f}")
        print(f"Hamming Loss: {hamming:.4f}")
        print("\nPer-Label F1 Scores:")
        for label, score in per_label_metrics.items():
            print(f"  {label}: {score:.4f}")
        print("="*50)
        
        return metrics

    def _create_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, output_path: str = None):
        """Create and save confusion matrix visualization.

        Args:
            y_true: True labels in binary format
            y_pred: Predicted labels in binary format
            output_path: Base path for saving the confusion matrix
        """
        print("Creating confusion matrix...")
        
        # Create confusion matrix for each label
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Confusion Matrix by Label', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, label_name in enumerate(self.label_names):
            if i < 4:  # Only plot first 4 labels
                cm = confusion_matrix(y_true[:, i], y_pred[:, i])
                
                # Create heatmap
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           ax=axes[i], cbar=False)
                axes[i].set_title(f'{label_name.title()}', fontweight='bold')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
                
                # Add accuracy text
                accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum() if cm.sum() > 0 else 0
                axes[i].text(0.5, -0.1, f'Accuracy: {accuracy:.3f}', 
                           ha='center', transform=axes[i].transAxes, fontweight='bold')
        
        plt.tight_layout()
        
        # Save confusion matrix
        if output_path:
            cm_path = output_path.replace('.csv', '_confusion_matrix.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {cm_path}")
        
        plt.show()

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
            predicted_labels.append('|'.join(labels) if labels else 'none')

        # Create results
        result = {
            'title': title,
            'abstract': abstract,
            'predicted_labels': predicted_labels[0],
            'prediction_matrix': predictions[0],
            'label_names': self.label_names
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
            predicted_labels.append('|'.join(labels) if labels else 'none')

        # Create results
        results = []
        for i, (title, abstract, pred_labels, pred_matrix) in enumerate(
            zip(titles, abstracts, predicted_labels, predictions)
        ):
            results.append({
                'title': title,
                'abstract': abstract,
                'predicted_labels': pred_labels,
                'prediction_matrix': pred_matrix,
                'label_names': self.label_names
            })

        print(f"Predictions completed for {len(titles)} articles")
        return results


def main():
    """Main function to demonstrate enhanced model testing."""
    
    # Initialize the enhanced tester
    tester = EnhancedModelTester()

    print("=" * 60)
    print("ENHANCED MODEL TESTING")
    print("=" * 60)

    # Example 1: Test with CSV file
    print("\n1. Testing with CSV file...")
    print("Note: Create a CSV file with columns: title, abstract, group")
    print("Example CSV format:")
    print("title;abstract;group")
    print("Cardiac arrhythmia detection;This study presents...;cardiovascular")
    print("Brain imaging analysis;This research investigates...;neurological")
    
    # Check if test CSV exists, if not create one
    test_csv_path = "input/test_data.csv"
    if not Path(test_csv_path).exists():
        print(f"\nCreating sample test CSV file: {test_csv_path}")
        sample_data = {
            'title': [
                "Cardiac arrhythmia detection using machine learning algorithms",
                "Neural network analysis of brain imaging data for Alzheimer's detection",
                "Liver function tests in patients with cardiovascular disease",
                "Novel chemotherapy agents for lung cancer treatment"
            ],
            'abstract': [
                "This study presents a novel approach for detecting cardiac arrhythmias using machine learning techniques.",
                "This research investigates the use of neural networks for analyzing brain imaging data to detect early signs of Alzheimer's disease.",
                "We examined liver function in 150 patients with cardiovascular disease.",
                "A new chemotherapy protocol was tested on 200 lung cancer patients."
            ],
            'group': [
                "cardiovascular",
                "neurological",
                "cardiovascular|hepatorenal",
                "oncological"
            ]
        }
        sample_df = pd.DataFrame(sample_data)
        sample_df.to_csv(test_csv_path, index=False, sep=';')
        print("Sample test CSV created successfully!")

    # Test with CSV file
    try:
        results = tester.predict_csv(test_csv_path, "output/csv_test_results.csv")
        
        print(f"\nResults Summary:")
        print(f"Total articles processed: {len(results['dataframe'])}")
        print(f"Predictions added to column: 'group_predicted'")
        
        if results['metrics']:
            print(f"Main Metric - Weighted F1 Score: {results['metrics']['weighted_f1']:.4f}")
        
        print(f"Results saved to: output/csv_test_results.csv")
        
    except Exception as e:
        print(f"Error testing with CSV: {e}")
        print("Continuing with single article test...")

    # Example 2: Test with a single article
    print("\n2. Testing with a single article...")
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
    print(f"Individual Predictions:")
    for i, label in enumerate(result["label_names"]):
        prediction = "YES" if result["prediction_matrix"][i] == 1 else "NO"
        print(f"  - {label}: {prediction}")

    print("\n" + "=" * 60)
    print("Enhanced testing completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()