"""
Cognitive Manifold Fingerprint Visualization (2x2 KDE Plots).

This script generates the cognitive state space visualization showing:
- 2x2 grid of kernel density estimates for each model
- Separation between hallucination and correct samples
- Visual "fingerprints" of model cognition

Usage:
    python visualizations/plot_manifold.py

    # With real data:
    python visualizations/plot_manifold.py \
        --data_dir results/ \
        --output cognitive_manifolds.pdf
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from pathlib import Path


def setup_style():
    """Configure high-quality research style."""
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']


def plot_2x2_fingerprints(model_data_dict: dict, output_path: str = "cognitive_manifolds.pdf"):
    """
    Plot 2x2 cognitive manifold fingerprints.

    Args:
        model_data_dict: Dictionary of {model_name: dataframe}
        output_path: Output PDF path
    """
    setup_style()

    # Colors
    correct_color = "#55a868"  # Green
    hallucination_color = "#c44e52"  # Red

    # Metrics
    metrics = [
        ('evidence_binary_entropy_mean', 'Perceptual Instability ($H_{Evi}$)'),
        ('logic_modality_diff', 'Inferential Conflict ($S_{Conf}$)'),
        ('final_answer_entropy', 'Decisional Ambiguity ($H_{Ans}$)')
    ]

    # Create figure
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    model_names = list(model_data_dict.keys())

    for idx, model_name in enumerate(model_names[:4]):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        df = model_data_dict[model_name]

        # Filter valid data
        df = df.dropna(subset=[m[0] for m in metrics] + ['is_hallucination'])

        # Split by hallucination
        correct = df[df['is_hallucination'] == 0]
        hallucination = df[df['is_hallucination'] == 1]

        # Plot KDE for each metric pair
        # Use first two metrics for 2D KDE
        x_col, x_label = metrics[0]
        y_col, y_label = metrics[1]

        # Plot hallucination first (on bottom)
        if len(hallucination) > 10:
            sns.kdeplot(
                data=hallucination, x=x_col, y=y_col,
                ax=ax, color=hallucination_color, levels=5,
                alpha=0.4, linewidths=2
            )

        # Plot correct (on top)
        if len(correct) > 10:
            sns.kdeplot(
                data=correct, x=x_col, y=y_col,
                ax=ax, color=correct_color, levels=5,
                alpha=0.4, linewidths=2
            )

        # Scatter for reference
        ax.scatter(correct[x_col].sample(min(50, len(correct))),
                  correct[y_col].sample(min(50, len(correct))),
                  c=correct_color, s=30, alpha=0.6, label='Correct', edgecolors='white')

        ax.scatter(hallucination[x_col].sample(min(50, len(hallucination))),
                  hallucination[y_col].sample(min(50, len(hallucination))),
                  c=hallucination_color, s=30, alpha=0.6, label='Hallucination', edgecolors='white')

        ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
        ax.set_title(f'({chr(97+idx)}) {model_name}', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)

        sns.despine(ax=ax)

    plt.suptitle('Cognitive State Space Fingerprints', fontsize=20, fontweight='bold', y=1.02)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Manifold fingerprints saved to {output_path}")


def generate_mock_data():
    """Generate mock data for testing."""
    np.random.seed(42)

    models = {
        "Idefics2": (0.947, 0.3, 1.5, 0.2, 1.0),
        "Llava-v1.6": (0.910, 0.4, 1.8, 0.3, 1.2),
        "Qwen2-VL": (0.776, 0.5, 2.0, 0.4, 1.5),
        "DeepSeek-VL": (0.798, 0.45, 1.9, 0.35, 1.3)
    }

    data_dict = {}

    for model, (auc, mu_h, sigma_h, mu_c, sigma_c) in models.items():
        n_samples = 500

        # Generate hallucination samples
        hallucination_df = pd.DataFrame({
            'evidence_binary_entropy_mean': np.random.normal(mu_h, sigma_h, n_samples // 2),
            'logic_modality_diff': np.random.normal(mu_h * 0.5, sigma_h * 0.5, n_samples // 2),
            'final_answer_entropy': np.random.normal(mu_h * 0.3, sigma_h * 0.3, n_samples // 2),
            'is_hallucination': 1
        })

        # Generate correct samples
        correct_df = pd.DataFrame({
            'evidence_binary_entropy_mean': np.random.normal(mu_c, sigma_c, n_samples // 2),
            'logic_modality_diff': np.random.normal(mu_c * 0.5, sigma_c * 0.5, n_samples // 2),
            'final_answer_entropy': np.random.normal(mu_c * 0.3, sigma_c * 0.3, n_samples // 2),
            'is_hallucination': 0
        })

        data_dict[model] = pd.concat([hallucination_df, correct_df], ignore_index=True)

    return data_dict


def load_results_from_dir(data_dir: str) -> dict:
    """Load results from directory."""
    data_path = Path(data_dir)
    model_data = {}

    for csv_file in data_path.glob('*/hallucination_metrics_full.csv'):
        model_name = csv_file.parent.name
        df = pd.read_csv(csv_file)
        model_data[model_name] = df

    return model_data


def main():
    parser = argparse.ArgumentParser(description='Plot cognitive manifold fingerprints')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory with extracted metrics')
    parser.add_argument('--output', type=str, default='cognitive_manifolds.pdf',
                       help='Output file path')

    args = parser.parse_args()

    if args.data_dir:
        model_data = load_results_from_dir(args.data_dir)
    else:
        model_data = generate_mock_data()

    plot_2x2_fingerprints(model_data, args.output)


if __name__ == "__main__":
    main()
