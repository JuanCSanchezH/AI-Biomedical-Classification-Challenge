"""Visualization utilities for model evaluation and results."""

import logging
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set style for better looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class ModelVisualizer:
    """Visualization class for model evaluation results."""

    def __init__(self, output_dir: str = "output"):
        """Initialize the visualizer.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir

    def create_comparison_charts(self, comparison_df: pd.DataFrame, save_path: str = None):
        """Create comparison charts for all 9 models as specified in requirements.

        Args:
            comparison_df: DataFrame with model comparison results
            save_path: Path to save the figure
        """
        if save_path is None:
            save_path = f"{self.output_dir}/model_comparison_charts.png"

        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Multi-Label Classification Model Comparison", fontsize=16, fontweight="bold")

        # Create model names for display
        model_names = [f"{row['strategy']} + {row['algorithm'].title()}" for _, row in comparison_df.iterrows()]

        # 1. Weighted F1 Score (top left)
        axes[0, 0].barh(model_names, comparison_df["weighted_f1"], color="skyblue", alpha=0.8)
        axes[0, 0].set_title("Weighted F1 Score", fontweight="bold")
        axes[0, 0].set_xlim(0, 1)
        axes[0, 0].set_xlabel("Weighted F1 Score")

        # Add value labels on bars
        for i, v in enumerate(comparison_df["weighted_f1"]):
            axes[0, 0].text(v + 0.01, i, f"{v:.3f}", va="center", fontweight="bold")

        # 2. Subset Accuracy (top right)
        axes[0, 1].barh(model_names, comparison_df["subset_accuracy"], color="lightgreen", alpha=0.8)
        axes[0, 1].set_title("Subset Accuracy", fontweight="bold")
        axes[0, 1].set_xlim(0, 1)
        axes[0, 1].set_xlabel("Subset Accuracy")

        # Add value labels on bars
        for i, v in enumerate(comparison_df["subset_accuracy"]):
            axes[0, 1].text(v + 0.01, i, f"{v:.3f}", va="center", fontweight="bold")

        # 3. Hamming Loss (bottom left)
        axes[1, 0].barh(model_names, comparison_df["hamming_loss"], color="salmon", alpha=0.8)
        axes[1, 0].set_title("Hamming Loss", fontweight="bold")
        axes[1, 0].set_xlim(0, 0.35)
        axes[1, 0].set_xlabel("Hamming Loss")

        # Add value labels on bars
        for i, v in enumerate(comparison_df["hamming_loss"]):
            axes[1, 0].text(v + 0.005, i, f"{v:.3f}", va="center", fontweight="bold")

        # 4. Metrics Heatmap (bottom right)
        metrics_for_heatmap = comparison_df[["weighted_f1", "subset_accuracy", "hamming_loss"]]
        sns.heatmap(
            metrics_for_heatmap.T,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            ax=axes[1, 1],
            cbar_kws={"label": "Metric Value"},
        )
        axes[1, 1].set_title("Metrics Heatmap", fontweight="bold")
        axes[1, 1].set_yticklabels(["Weighted F1", "Subset Accuracy", "Hamming Loss"])
        axes[1, 1].set_xticklabels(model_names, rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"Comparison charts saved to {save_path}")

    def create_best_model_chart(self, best_model_metrics: Dict[str, float], model_name: str, save_path: str = None):
        """Create chart for the best model metrics.

        Args:
            best_model_metrics: Dictionary with metrics for the best model
            model_name: Name of the best model
            save_path: Path to save the figure
        """
        if save_path is None:
            save_path = f"{self.output_dir}/best_model_metrics.png"

        # Extract metrics for plotting
        metrics = ["weighted_f1", "micro_f1", "macro_f1", "subset_accuracy"]
        values = [best_model_metrics[metric] for metric in metrics]
        labels = ["Weighted F1", "Micro F1", "Macro F1", "Subset Accuracy"]

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, values, color=["skyblue", "lightgreen", "gold", "salmon"], alpha=0.8)

        # Add value labels on bars
        for bar, value in zip(bars, values, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_title(f"Best Model Performance: {model_name}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"Best model chart saved to {save_path}")

    def create_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str], save_path: str = None
    ):
        """Create confusion matrix for the best model.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            label_names: Names of the labels
            save_path: Path to save the figure
        """
        if save_path is None:
            save_path = f"{self.output_dir}/confusion_matrix.png"

        # Calculate confusion matrix for each label
        MAX_LABELS_TO_PLOT = 4
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Confusion Matrix by Label", fontsize=16, fontweight="bold")

        axes = axes.flatten()

        for i, label_name in enumerate(label_names):
            if i < MAX_LABELS_TO_PLOT:  # Only plot first 4 labels
                cm = confusion_matrix(y_true[:, i], y_pred[:, i])

                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i], cbar=False)
                axes[i].set_title(f"{label_name.title()}", fontweight="bold")
                axes[i].set_xlabel("Predicted")
                axes[i].set_ylabel("Actual")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"Confusion matrix saved to {save_path}")

    def create_comparison_table(self, comparison_df: pd.DataFrame, save_path: str = None):
        """Create and save comparison table.

        Args:
            comparison_df: DataFrame with model comparison results
            save_path: Path to save the table
        """
        if save_path is None:
            save_path = f"{self.output_dir}/model_comparison_table.csv"

        # Format the DataFrame for better display
        display_df = comparison_df.copy()
        display_df["algorithm"] = display_df["algorithm"].str.title()
        display_df["strategy"] = display_df["strategy"].str.upper()

        # Round numeric columns
        numeric_cols = ["weighted_f1", "micro_f1", "macro_f1", "subset_accuracy", "hamming_loss"]
        for col in numeric_cols:
            display_df[col] = display_df[col].round(4)

        # Save to CSV
        display_df.to_csv(save_path, index=False)

        logging.info(f"Comparison table saved to {save_path}")

        return display_df

    def create_data_exploration_charts(self, data_info: Dict[str, Any], save_path: str = None):
        """Create data exploration charts.

        Args:
            data_info: Dictionary with data information
            save_path: Path to save the figure
        """
        if save_path is None:
            save_path = f"{self.output_dir}/data_exploration.png"

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Dataset Exploration", fontsize=16, fontweight="bold")

        # 1. Group distribution
        group_dist = data_info["group_distribution"]
        top_groups = dict(sorted(group_dist.items(), key=lambda x: x[1], reverse=True)[:10])

        axes[0, 0].barh(list(top_groups.keys()), list(top_groups.values()), color="skyblue", alpha=0.8)
        axes[0, 0].set_title("Top 10 Group Distribution", fontweight="bold")
        axes[0, 0].set_xlabel("Count")

        # 2. Title length distribution
        title_stats = data_info["title_length_stats"]
        axes[0, 1].hist([title_stats["mean"], title_stats["median"]], bins=20, alpha=0.7, label=["Mean", "Median"])
        axes[0, 1].set_title("Title Length Statistics", fontweight="bold")
        axes[0, 1].set_xlabel("Length (characters)")
        axes[0, 1].legend()

        # 3. Abstract length distribution
        abstract_stats = data_info["abstract_length_stats"]
        axes[1, 0].hist(
            [abstract_stats["mean"], abstract_stats["median"]], bins=20, alpha=0.7, label=["Mean", "Median"]
        )
        axes[1, 0].set_title("Abstract Length Statistics", fontweight="bold")
        axes[1, 0].set_xlabel("Length (characters)")
        axes[1, 0].legend()

        # 4. Dataset overview
        overview_data = {"Total Samples": data_info["total_samples"], "Unique Groups": data_info["unique_groups"]}
        axes[1, 1].bar(overview_data.keys(), overview_data.values(), color=["lightgreen", "gold"], alpha=0.8)
        axes[1, 1].set_title("Dataset Overview", fontweight="bold")
        axes[1, 1].set_ylabel("Count")

        # Add value labels
        for i, v in enumerate(overview_data.values()):
            axes[1, 1].text(i, v + 50, str(v), ha="center", va="bottom", fontweight="bold")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"Data exploration charts saved to {save_path}")


def setup_plotting_style():
    """Setup consistent plotting style."""
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 16
