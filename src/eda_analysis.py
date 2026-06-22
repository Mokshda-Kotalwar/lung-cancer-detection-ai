"""
Exploratory Data Analysis (EDA) Module for Lung Cancer Dataset
Generates statistics, histograms, pie charts, image size metrics, and imbalance reports.
Author: Senior AI Engineer & Medical Imaging Expert
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


class LungDatasetEDA:
    """
    Performs complete exploratory data analysis for CT slices, resolutions,
    and class imbalances.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Style configurations
        sns.set_theme(style="darkgrid")
        plt.rcParams.update({
            'figure.facecolor': '#1e1e1e',
            'axes.facecolor': '#1e1e1e',
            'text.color': '#ffffff',
            'axes.labelcolor': '#ffffff',
            'xtick.color': '#ffffff',
            'ytick.color': '#ffffff',
            'axes.edgecolor': '#444444'
        })

    def analyze_class_distribution(self, labels: List[str]) -> Dict[str, Any]:
        """
        Analyze and plot the frequency of each class in the dataset.
        """
        df = pd.DataFrame(labels, columns=["Class"])
        counts = df["Class"].value_counts()
        percentages = df["Class"].value_counts(normalize=True) * 100
        
        # 1. Bar Plot (Class Distribution)
        fig, ax = plt.subplots(figsize=(8, 6))
        palette = sns.color_palette("muted", len(counts))
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=palette)
        ax.set_title("Class Distribution (Volume Counts)", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Diagnostic Class", fontsize=12)
        ax.set_ylabel("Number of Scans", fontsize=12)
        
        for i, val in enumerate(counts.values):
            ax.text(i, val + (max(counts.values) * 0.01), f"{val} ({percentages.iloc[i]:.1f}%)", 
                    ha='center', va='bottom', color='white', fontweight='bold')
            
        fig.tight_layout()
        bar_path = self.output_dir / "class_distribution_bar.png"
        fig.savefig(bar_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        
        # 2. Pie Chart
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=140, 
               colors=palette, textprops={'color': 'white', 'weight': 'bold', 'fontsize': 12})
        ax.set_title("Class Proportion (%)", fontsize=14, fontweight='bold', pad=15)
        
        fig.tight_layout()
        pie_path = self.output_dir / "class_distribution_pie.png"
        fig.savefig(pie_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()

        # Calculate imbalance metric (Shannon Entropy of distribution)
        # H = - sum(p_i * log2(p_i))
        p = counts.values / counts.sum()
        entropy = -np.sum(p * np.log2(p))
        max_entropy = np.log2(len(counts))
        balance_index = entropy / max_entropy if max_entropy > 0 else 1.0

        stats = {
            "counts": counts.to_dict(),
            "percentages": percentages.to_dict(),
            "entropy": float(entropy),
            "balance_index": float(balance_index),
            "total_samples": int(counts.sum()),
            "bar_plot_path": str(bar_path),
            "pie_plot_path": str(pie_path)
        }
        
        logger.info(f"Class Distribution: {stats['counts']}")
        logger.info(f"Dataset Balance Index (0-1): {stats['balance_index']:.3f} (1.0 = Perfect Balance)")
        
        return stats

    def analyze_pixel_histograms(self, sample_images: List[np.ndarray]) -> str:
        """
        Analyze Hounsfield scale or raw intensity distributions.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, img in enumerate(sample_images[:3]):
            sns.kdeplot(img.flatten(), ax=ax, label=f"Scan Sample {i+1}", fill=True, alpha=0.15)
            
        ax.set_title("Pixel Value Intensity Histogram (HU Scales)", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Intensity / Hounsfield Units (HU)", fontsize=12)
        ax.set_ylabel("Density Distribution", fontsize=12)
        ax.legend(facecolor='#1e1e1e', edgecolor='#444444')
        
        fig.tight_layout()
        hist_path = self.output_dir / "pixel_intensity_histogram.png"
        fig.savefig(hist_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        
        return str(hist_path)

    def analyze_image_sizes(self, dimensions: List[Tuple[int, int]]) -> Dict[str, Any]:
        """
        Plot and report image resolutions.
        """
        df = pd.DataFrame(dimensions, columns=["Width", "Height"])
        df["Aspect_Ratio"] = df["Width"] / df["Height"]
        
        unique_resolutions = df.groupby(["Width", "Height"]).size().reset_index(name="Count")
        
        # Resolution scatterplot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=unique_resolutions, x="Width", y="Height", size="Count", 
                        sizes=(40, 400), ax=ax, color="#00adb5", alpha=0.8)
        ax.set_title("Resolution Distribution of Slice Images", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Width (pixels)", fontsize=12)
        ax.set_ylabel("Height (pixels)", fontsize=12)
        
        fig.tight_layout()
        size_path = self.output_dir / "resolution_scatter.png"
        fig.savefig(size_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        
        stats = {
            "unique_sizes": unique_resolutions.to_dict(orient="records"),
            "aspect_ratio_mean": float(df["Aspect_Ratio"].mean()),
            "plot_path": str(size_path)
        }
        
        return stats

    def run_detection_alert(self, balance_index: float) -> str:
        """Evaluate dataset skewness."""
        if balance_index < 0.6:
            return "WARNING: Severe imbalance detected. Use Class Weights, oversampling, or Focal Loss."
        elif balance_index < 0.85:
            return "NOTE: Moderate imbalance detected. Monitor validation curves."
        return "SUCCESS: Dataset is well balanced."


# Standalone runner for validation
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import config, DATA_DIR, OUTPUTS_DIR

    
    eda = LungDatasetEDA(OUTPUTS_DIR / "eda")
    
    # 1. Mock dataset characteristics
    # Suppose LIDC/IQ datasets have:
    mock_labels = ["Malignant"] * 30 + ["Benign"] * 60 + ["Normal"] * 110
    mock_dimensions = [(512, 512)] * 180 + [(768, 768)] * 15 + [(1024, 1024)] * 5
    
    # Generate some dummy CT pixel arrays
    mock_scans = [
        np.random.normal(-600, 300, (512, 512)),   # Lung window centered
        np.random.normal(40, 200, (512, 512)),     # Soft tissue centered
        np.random.normal(-1000, 50, (512, 512))    # Air/Empty centered
    ]
    
    print("Running exploratory data analysis...")
    class_stats = eda.analyze_class_distribution(mock_labels)
    size_stats = eda.analyze_image_sizes(mock_dimensions)
    hist_path = eda.analyze_pixel_histograms(mock_scans)
    alert = eda.run_detection_alert(class_stats["balance_index"])
    
    print("\n--- EDA Insights ---")
    print(f"Total images analyzed: {class_stats['total_samples']}")
    print(f"Balance Index: {class_stats['balance_index']:.3f}")
    print(f"Status: {alert}")
    print(f"Saved Bar chart: {class_stats['bar_plot_path']}")
    print(f"Saved Pie chart: {class_stats['pie_plot_path']}")
    print(f"Saved Histogram: {hist_path}")
    print(f"Saved Dimension Plot: {size_stats['plot_path']}")
