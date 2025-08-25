"""
Main script for medical article multi-label classification challenge.
Implements the complete pipeline: data processing, feature engineering, 
model training, evaluation, and output generation.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing import DataProcessor
from feature_engineering import FeatureEngineer
from models import ModelTrainer
from evaluation import MultiLabelEvaluator
from utils import (
    create_output_directory, 
    save_predictions, 
    plot_metrics_comparison,
    print_dataset_summary
)


def main():
    """Main function to run the complete pipeline."""
    
    print("=== MEDICAL ARTICLE CLASSIFICATION CHALLENGE ===")
    print("Starting the complete pipeline...\n")
    
    # Define paths
    input_path = "input/challenge_data.csv"
    output_path = "output"
    
    # Create output directory
    create_output_directory(output_path)
    
    # Step 1: Data Processing
    print("STEP 1: DATA PROCESSING")
    print("=" * 50)
    
    data_processor = DataProcessor(input_path)
    
    # Load and analyze data
    data = data_processor.load_data()
    if data is None:
        print("Error: Could not load data. Exiting.")
        return
    
    # Print dataset summary
    print_dataset_summary(data)
    
    # Prepare data for multi-label classification
    X_train, X_test, y_train, y_test, label_names = data_processor.prepare_data()
    
    print(f"\nData preparation completed:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Number of labels: {len(label_names)}")
    print(f"  Labels: {label_names}")
    
    # Step 2: Feature Engineering
    print("\nSTEP 2: FEATURE ENGINEERING")
    print("=" * 50)
    
    feature_engineer = FeatureEngineer(max_features=3000, n_components=500)
    
    # Create TF-IDF features
    print("Creating TF-IDF features...")
    X_train_tfidf, X_test_tfidf = feature_engineer.create_tfidf_features(
        X_train, X_test, fit=True
    )
    
    print(f"Feature engineering completed:")
    print(f"  Training features: {X_train_tfidf.shape}")
    print(f"  Test features: {X_test_tfidf.shape}")
    
    # Step 3: Model Training (All 9 combinations)
    print("\nSTEP 3: MODEL TRAINING")
    print("=" * 50)
    print("Training all 9 model combinations (3 strategies × 3 algorithms)...")
    
    model_trainer = ModelTrainer()
    
    # Train all models
    results = model_trainer.train_all_models(
        X_train_tfidf, y_train, X_test_tfidf, y_test
    )
    
    # Step 4: Model Evaluation
    print("\nSTEP 4: MODEL EVALUATION")
    print("=" * 50)
    
    evaluator = MultiLabelEvaluator()
    
    # Evaluate all models
    print("Calculating evaluation metrics for all models...")
    metrics_df = evaluator.evaluate_all_models(results)
    
    # Create comparison table
    comparison_table = evaluator.create_comparison_table(metrics_df)
    
    # Print results
    print("\nModel Performance Comparison (sorted by Weighted F1):")
    print("-" * 80)
    print(comparison_table.to_string(index=False))
    
    # Step 5: Select Best Model
    print("\nSTEP 5: SELECTING BEST MODEL")
    print("=" * 50)
    
    best_model_name = evaluator.get_best_model(metrics_df, metric='f1_weighted')
    
    if best_model_name:
        print(f"Best model based on Weighted F1: {best_model_name}")
        
        # Get best model details
        best_result = results[best_model_name]
        best_metrics = metrics_df[metrics_df['model_name'] == best_model_name].iloc[0]
        
        print(f"\nBest model performance:")
        print(f"  Weighted F1: {best_metrics['f1_weighted']:.4f}")
        print(f"  Subset Accuracy: {best_metrics['subset_accuracy']:.4f}")
        print(f"  Hamming Loss: {best_metrics['hamming_loss']:.4f}")
        
        # Print detailed report for best model
        evaluator.print_detailed_report(
            best_result['true_labels'], 
            best_result['predictions'], 
            best_model_name, 
            label_names
        )
    else:
        print("No valid model found!")
        return
    
    # Step 6: Generate Output Files
    print("\nSTEP 6: GENERATING OUTPUT FILES")
    print("=" * 50)
    
    # Save predictions
    predictions_path = os.path.join(output_path, "predictions.csv")
    save_predictions(
        X_test, y_test, best_result['predictions'], 
        label_names, predictions_path
    )
    
    # Save comparison table
    comparison_path = os.path.join(output_path, "results_comparison.csv")
    comparison_table.to_csv(comparison_path, index=False)
    print(f"Comparison table saved to {comparison_path}")
    
    # Create visualization
    plot_path = os.path.join(output_path, "model_comparison.png")
    plot_metrics_comparison(metrics_df, plot_path)
    
    # Step 7: Final Summary
    print("\nSTEP 7: FINAL SUMMARY")
    print("=" * 50)
    
    print("Challenge completed successfully!")
    print(f"\nOutput files created in '{output_path}':")
    print(f"  - predictions.csv: Test dataset with predictions")
    print(f"  - results_comparison.csv: Model comparison table")
    print(f"  - model_comparison.png: Performance visualization")
    
    print(f"\nBest performing model: {best_model_name}")
    print(f"Strategy: {best_model_name.split('_')[0]}")
    print(f"Algorithm: {best_model_name.split('_')[1]}")
    
    print(f"\nFinal metrics:")
    print(f"  Weighted F1 Score: {best_metrics['f1_weighted']:.4f}")
    print(f"  Subset Accuracy: {best_metrics['subset_accuracy']:.4f}")
    print(f"  Hamming Loss: {best_metrics['hamming_loss']:.4f}")
    
    print("\n=== CHALLENGE COMPLETED ===")


if __name__ == "__main__":
    main() 