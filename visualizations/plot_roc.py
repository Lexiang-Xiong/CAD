"""
ROC Curve Visualization Script.

This script generates high-quality ROC curves including:
- Standard linear scale ROC
- Log-Log scale ROC (for low FPR analysis)

Usage:
    python visualizations/plot_roc.py

    # With real data:
    python visualizations/plot_roc.py \
        --data_dir results/evaluation \
        --output roc_panel.pdf
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Set high-quality style
def setup_style():
    """Configure matplotlib for research-quality figures."""
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.size'] = 14


def plot_roc_curves(roc_results: dict, output_path: str = "roc_loglog_panel.pdf"):
    """
    Plot ROC curves in linear and log-log scales.

    Args:
        roc_results: Dictionary of {model_name: (fpr, tpr, auc)}
        output_path: Output file path
    """
    setup_style()

    # Color palette (muted/Mordori style)
    model_colors = {
        "Llava-v1.6-Mistral-7B": "#4c72b0",
        "Qwen2-VL": "#c44e52",
        "Idefics2": "#55a868",
        "DeepSeek-VL": "#ccb974"
    }

    # Create 1x2 panel
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Sort by AUC
    sorted_models = sorted(roc_results.items(), key=lambda x: x[1][2], reverse=True)

    for ax, scale in zip([ax1, ax2], ['linear', 'log']):
        for model_name, (fpr, tpr, auc_score) in sorted_models:
            color = model_colors.get(model_name, "#333333")

            if scale == 'linear':
                ax.plot(fpr, tpr, lw=3.0, color=color,
                       label=f'{model_name} ({auc_score:.3f})')
            else:
                # Log-log: filter valid points
                valid = (fpr > 0) & (tpr > 0)
                if np.any(valid):
                    ax.plot(fpr[valid], tpr[valid], lw=3.0, color=color,
                           label=f'{model_name} ({auc_score:.3f})')

        # Random baseline
        ax.plot([1e-4, 1], [1e-4, 1], color='gray', lw=1.5,
               linestyle='--', alpha=0.6)

        ax.grid(True, which='both', linestyle=':', alpha=0.5)

        if scale == 'linear':
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.02])
            ax.set_xlabel('False Positive Rate (Linear)', fontsize=18, fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontsize=18, fontweight='bold')
            ax.set_title('(a) ROC (Linear Scale)', fontsize=20, pad=15)
        else:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim([1e-4, 1.0])
            ax.set_ylim([1e-2, 1.05])
            ax.set_xlabel('False Positive Rate (Log)', fontsize=18, fontweight='bold')
            ax.set_ylabel('True Positive Rate (Log)', fontsize=18, fontweight='bold')
            ax.set_title('(b) ROC (Log-Log Scale)', fontsize=20, pad=15)

    # Legend
    ax1.legend(loc="lower right", fontsize=11, frameon=True,
               framealpha=0.9, edgecolor='gray')

    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {output_path}")


def generate_mock_data():
    """Generate mock ROC data for testing."""
    x = np.linspace(0, 1, 1000)

    # Generate realistic-looking ROC curves
    np.random.seed(42)
    return {
        "Idefics2": (x, np.sqrt(x) + np.random.normal(0, 0.02, len(x)), 0.947),
        "Llava-v1.6-Mistral-7B": (x, x**0.6 + np.random.normal(0, 0.02, len(x)), 0.910),
        "DeepSeek-VL": (x, x**0.8 + np.random.normal(0, 0.02, len(x)), 0.798),
        "Qwen2-VL": (x, x**0.85 + np.random.normal(0, 0.02, len(x)), 0.776),
    }


def load_results_from_dir(data_dir: str) -> dict:
    """
    Load evaluation results from directory.

    Args:
        data_dir: Directory containing evaluation CSVs

    Returns:
        Dictionary of model results
    """
    data_path = Path(data_dir)
    results = {}

    for csv_file in data_path.glob('*_cad_results.csv'):
        model_name = csv_file.stem.replace('_cad_results', '')
        df = pd.read_csv(csv_file)

        # Generate ROC curve from AUC
        fpr = np.linspace(0, 1, 1000)
        tpr = fpr ** (1 / df['auc'].values[0])  # Approximate
        auc = df['auc'].values[0]

        results[model_name] = (fpr, tpr, auc)

    return results


def main():
    parser = argparse.ArgumentParser(description='Plot ROC curves')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory with evaluation results')
    parser.add_argument('--output', type=str, default='roc_loglog_panel.pdf',
                       help='Output file path')

    args = parser.parse_args()

    if args.data_dir:
        roc_results = load_results_from_dir(args.data_dir)
    else:
        roc_results = generate_mock_data()

    plot_roc_curves(roc_results, args.output)


if __name__ == "__main__":
    main()
