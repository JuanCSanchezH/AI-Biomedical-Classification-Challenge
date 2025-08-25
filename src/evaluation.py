"""
Evaluation module for multi-label classification.
Implements proper multi-label metrics: Hamming Loss, F1 scores, Subset Accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import hamming_loss, f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class MultiLabelEvaluator:
    """Evaluator for multi-label classification results."""
    
    def __init__(self):
        """Initialize MultiLabelEvaluator."""
        self.metrics = {}
    
    def calculate_metrics(self, y_true, y_pred, model_name=""):
        """
        Calculate all multi-label classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name (str): Name of the model for identification
            
        Returns:
            dict: Dictionary containing all metrics
        """
        if y_true is None or y_pred is None:
            return {
                'model_name': model_name,
                'hamming_loss': None,
                'f1_micro': None,
                'f1_macro': None,
                'f1_weighted': None,
                'subset_accuracy': None,
                'error': 'No predictions available'
            }
        
        try:
            # Convert sparse matrices to dense arrays first
            if hasattr(y_true, 'toarray'):
                y_true = y_true.toarray()
            if hasattr(y_pred, 'toarray'):
                y_pred = y_pred.toarray()
            
            # Convert to numpy arrays if needed
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Calculate metrics
            hamming = hamming_loss(y_true, y_pred)
            f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Subset accuracy (exact match)
            subset_acc = accuracy_score(y_true, y_pred)
            
            metrics = {
                'model_name': model_name,
                'hamming_loss': hamming,
                'f1_micro': f1_micro,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'subset_accuracy': subset_acc
            }
            
            return metrics
            
        except Exception as e:
            return {
                'model_name': model_name,
                'hamming_loss': None,
                'f1_micro': None,
                'f1_macro': None,
                'f1_weighted': None,
                'subset_accuracy': None,
                'error': str(e)
            }
    
    def evaluate_all_models(self, results_dict):
        """
        Evaluate all models in the results dictionary.
        
        Args:
            results_dict (dict): Dictionary containing model results
            
        Returns:
            pd.DataFrame: DataFrame with all metrics for comparison
        """
        all_metrics = []
        
        for model_name, result in results_dict.items():
            y_true = result.get('true_labels')
            y_pred = result.get('predictions')
            
            metrics = self.calculate_metrics(y_true, y_pred, model_name)
            all_metrics.append(metrics)
        
        # Create DataFrame
        metrics_df = pd.DataFrame(all_metrics)
        
        # Sort by weighted F1 score (best first)
        if 'f1_weighted' in metrics_df.columns:
            metrics_df = metrics_df.sort_values('f1_weighted', ascending=False, na_position='last')
        
        return metrics_df
    
    def create_comparison_table(self, metrics_df):
        """
        Create a formatted comparison table.
        
        Args:
            metrics_df (pd.DataFrame): DataFrame with metrics
            
        Returns:
            pd.DataFrame: Formatted comparison table
        """
        # Create rank column
        metrics_df['rank'] = range(1, len(metrics_df) + 1)
        
        # Select and rename columns for output
        comparison_table = metrics_df[['rank', 'model_name', 'f1_weighted', 
                                     'subset_accuracy', 'hamming_loss']].copy()
        
        # Extract strategy and algorithm from model name
        comparison_table['strategy'] = comparison_table['model_name'].str.split('_').str[0]
        comparison_table['algorithm'] = comparison_table['model_name'].str.split('_').str[1]
        
        # Rename columns for clarity
        comparison_table.columns = ['Rank', 'Model', 'Weighted F1', 
                                  'Subset Accuracy', 'Hamming Loss', 'Strategy', 'Algorithm']
        
        # Format numeric columns
        numeric_cols = ['Weighted F1', 'Subset Accuracy', 'Hamming Loss']
        for col in numeric_cols:
            if col in comparison_table.columns:
                comparison_table[col] = comparison_table[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                )
        
        return comparison_table
    
    def get_best_model(self, metrics_df, metric='f1_weighted'):
        """
        Get the best model based on a specific metric.
        
        Args:
            metrics_df (pd.DataFrame): DataFrame with metrics
            metric (str): Metric to use for comparison
            
        Returns:
            str: Name of the best model
        """
        if metric not in metrics_df.columns:
            raise ValueError(f"Metric {metric} not found in metrics DataFrame")
        
        # Filter out models with errors
        valid_models = metrics_df[metrics_df[metric].notna()]
        
        if valid_models.empty:
            return None
        
        # Get the best model
        best_model = valid_models.loc[valid_models[metric].idxmax()]
        return best_model['model_name']
    
    def print_detailed_report(self, y_true, y_pred, model_name, label_names=None):
        """
        Print detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name (str): Name of the model
            label_names (list): Names of the labels
        """
        print(f"\n=== Detailed Report for {model_name} ===")
        
        if y_true is None or y_pred is None:
            print("No predictions available for detailed report.")
            return
        
        try:
            # Basic metrics
            metrics = self.calculate_metrics(y_true, y_pred, model_name)
            print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
            print(f"Micro F1: {metrics['f1_micro']:.4f}")
            print(f"Macro F1: {metrics['f1_macro']:.4f}")
            print(f"Weighted F1: {metrics['f1_weighted']:.4f}")
            print(f"Subset Accuracy: {metrics['subset_accuracy']:.4f}")
            
            # Per-label metrics
            if label_names is not None:
                print("\nPer-label F1 scores:")
                for i, label in enumerate(label_names):
                    try:
                        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
                        print(f"  {label}: {f1:.4f}")
                    except Exception as e:
                        print(f"  {label}: Error calculating F1 - {e}")
            
        except Exception as e:
            print(f"Error generating detailed report: {e}")
    
    def save_results(self, metrics_df, output_path):
        """
        Save results to CSV file.
        
        Args:
            metrics_df (pd.DataFrame): DataFrame with metrics
            output_path (str): Path to save the results
        """
        try:
            metrics_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error saving results: {e}") 