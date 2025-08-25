"""
Utilities module for medical article classification.
Contains helper functions for visualization, data handling, and output generation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
import os
import warnings
warnings.filterwarnings('ignore')


def create_output_directory(output_path):
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_path (str): Path to the output directory
    """
    os.makedirs(output_path, exist_ok=True)
    print(f"Output directory created/verified: {output_path}")


def save_predictions(X_test, y_test, y_pred, label_names, output_path):
    """
    Save predictions to CSV file with original data and predictions.
    
    Args:
        X_test: Test features (text data)
        y_test: True labels
        y_pred: Predicted labels
        label_names: Names of the labels
        output_path (str): Path to save the predictions
    """
    try:
        # Convert predictions back to label strings
        mlb = MultiLabelBinarizer()
        mlb.classes_ = label_names
        
        # Convert binary predictions to label strings
        predicted_labels = []
        for pred in y_pred:
            # Convert sparse matrix to dense if needed
            if hasattr(pred, 'toarray'):
                pred = pred.toarray()
            labels = mlb.inverse_transform([pred])[0]
            predicted_labels.append('|'.join(labels) if labels else '')
        
        # Create DataFrame
        results_df = pd.DataFrame({
            'text': X_test,
            'true_labels': [mlb.inverse_transform([y])[0] for y in y_test],
            'predicted_labels': predicted_labels
        })
        
        # Convert list to string for true_labels
        results_df['true_labels'] = results_df['true_labels'].apply(lambda x: '|'.join(x) if x else '')
        
        # Save to CSV
        results_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        
        return results_df
        
    except Exception as e:
        print(f"Error saving predictions: {e}")
        return None


def plot_metrics_comparison(metrics_df, output_path=None):
    """
    Create visualization of model comparison metrics.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame with metrics
        output_path (str): Path to save the plot (optional)
    """
    try:
        # Filter out models with errors
        valid_metrics = metrics_df[metrics_df['f1_weighted'].notna()].copy()
        
        if valid_metrics.empty:
            print("No valid metrics to plot")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Extract model names for plotting
        model_names = valid_metrics['model_name'].str.replace('_', '\n')
        
        # Plot 1: Weighted F1 Score
        axes[0, 0].bar(range(len(model_names)), valid_metrics['f1_weighted'])
        axes[0, 0].set_title('Weighted F1 Score')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].set_xticks(range(len(model_names)))
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Subset Accuracy
        axes[0, 1].bar(range(len(model_names)), valid_metrics['subset_accuracy'])
        axes[0, 1].set_title('Subset Accuracy')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_xticks(range(len(model_names)))
        axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Hamming Loss
        axes[1, 0].bar(range(len(model_names)), valid_metrics['hamming_loss'])
        axes[1, 0].set_title('Hamming Loss')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_xticks(range(len(model_names)))
        axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Micro vs Macro F1
        x_pos = np.arange(len(model_names))
        width = 0.35
        
        axes[1, 1].bar(x_pos - width/2, valid_metrics['f1_micro'], width, label='Micro F1')
        axes[1, 1].bar(x_pos + width/2, valid_metrics['f1_macro'], width, label='Macro F1')
        axes[1, 1].set_title('Micro vs Macro F1 Score')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error creating metrics plot: {e}")


def analyze_label_distribution(data, group_column='group'):
    """
    Analyze the distribution of labels in the dataset.
    
    Args:
        data (pd.DataFrame): Input dataset
        group_column (str): Name of the column containing labels
        
    Returns:
        dict: Analysis results
    """
    try:
        # Split labels and count
        all_labels = []
        for labels_str in data[group_column]:
            if pd.notna(labels_str):
                labels = labels_str.split('|')
                all_labels.extend(labels)
        
        # Count unique labels
        label_counts = pd.Series(all_labels).value_counts()
        
        # Count articles per label combination
        combination_counts = data[group_column].value_counts()
        
        # Calculate statistics
        analysis = {
            'total_articles': len(data),
            'unique_labels': len(label_counts),
            'label_counts': label_counts,
            'combination_counts': combination_counts,
            'avg_labels_per_article': len(all_labels) / len(data),
            'single_label_articles': len(data[data[group_column].str.count('\\|') == 0]),
            'multi_label_articles': len(data[data[group_column].str.count('\\|') > 0])
        }
        
        return analysis
        
    except Exception as e:
        print(f"Error analyzing label distribution: {e}")
        return None


def print_dataset_summary(data, group_column='group'):
    """
    Print a summary of the dataset.
    
    Args:
        data (pd.DataFrame): Input dataset
        group_column (str): Name of the column containing labels
    """
    print("=== DATASET SUMMARY ===")
    print(f"Total articles: {len(data)}")
    print(f"Unique titles: {data['title'].nunique()}")
    print(f"Unique abstracts: {data['abstract'].nunique()}")
    
    # Title length statistics
    title_lengths = data['title'].str.len()
    print(f"\nTitle length statistics:")
    print(f"  Mean: {title_lengths.mean():.1f} characters")
    print(f"  Median: {title_lengths.median():.1f} characters")
    print(f"  Min: {title_lengths.min()} characters")
    print(f"  Max: {title_lengths.max()} characters")
    
    # Abstract length statistics
    abstract_lengths = data['abstract'].str.len()
    print(f"\nAbstract length statistics:")
    print(f"  Mean: {abstract_lengths.mean():.1f} characters")
    print(f"  Median: {abstract_lengths.median():.1f} characters")
    print(f"  Min: {abstract_lengths.min()} characters")
    print(f"  Max: {abstract_lengths.max()} characters")
    
    # Label analysis
    analysis = analyze_label_distribution(data, group_column)
    if analysis:
        print(f"\nLabel analysis:")
        print(f"  Unique labels: {analysis['unique_labels']}")
        print(f"  Average labels per article: {analysis['avg_labels_per_article']:.2f}")
        print(f"  Single-label articles: {analysis['single_label_articles']}")
        print(f"  Multi-label articles: {analysis['multi_label_articles']}")
        
        print(f"\nTop 10 label combinations:")
        for i, (combination, count) in enumerate(analysis['combination_counts'].head(10).items()):
            print(f"  {i+1}. {combination}: {count} articles")


def format_model_name(model_name):
    """
    Format model name for display.
    
    Args:
        model_name (str): Raw model name
        
    Returns:
        str: Formatted model name
    """
    if '_' in model_name:
        strategy, algorithm = model_name.split('_', 1)
        
        strategy_names = {
            'BR': 'Binary Relevance',
            'CC': 'Classifier Chains',
            'LP': 'Label Powerset'
        }
        
        algorithm_names = {
            'LR': 'Logistic Regression',
            'XGB': 'XGBoost',
            'SVM': 'SVM'
        }
        
        strategy_display = strategy_names.get(strategy, strategy)
        algorithm_display = algorithm_names.get(algorithm, algorithm)
        
        return f"{strategy_display} + {algorithm_display}"
    
    return model_name 