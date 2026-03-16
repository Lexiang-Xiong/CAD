"""
Ablation Study Script: Evaluate contribution of each metric component.

This script:
1. Evaluates CAD performance with each metric individually
2. Evaluates CAD with pairwise combinations
3. Evaluates CAD with all three metrics
4. Computes synergy gain and component importance

Usage:
    python scripts/evaluation/run_ablation.py \
        --input_file results/idefics2/hallucination_metrics_full.csv \
        --model_name Idefics2
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.detector.cad_gmm import CognitiveAnomalyDetector


def load_data(csv_path: str):
    """Load and preprocess data."""
    df = pd.read_csv(csv_path)
    df = df[df['status'] == 'success']
    df = df.dropna(subset=['evidence_binary_entropy_mean', 'logic_modality_diff',
                           'final_answer_entropy', 'is_hallucination'])

    return df


def evaluate_single_metric(df, metric_col, n_components=5, test_size=0.3):
    """Evaluate CAD with a single metric."""
    X = df[[metric_col]].values
    y = df['is_hallucination'].values.astype(int)

    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    X_nominal = X_pool[y_pool == 0]

    if len(X_nominal) < n_components:
        return None

    detector = CognitiveAnomalyDetector(n_components=n_components)
    detector.fit(X_nominal)
    scores = detector.predict_surprisal(X_test)

    return roc_auc_score(y_test, scores)


def evaluate_pair_metrics(df, metric_cols, n_components=5, test_size=0.3):
    """Evaluate CAD with a pair of metrics."""
    X = df[metric_cols].values
    y = df['is_hallucination'].values.astype(int)

    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    X_nominal = X_pool[y_pool == 0]

    if len(X_nominal) < n_components:
        return None

    detector = CognitiveAnomalyDetector(n_components=n_components)
    detector.fit(X_nominal)
    scores = detector.predict_surprisal(X_test)

    return roc_auc_score(y_test, scores)


def evaluate_all_metrics(df, n_components=5, test_size=0.3):
    """Evaluate CAD with all three metrics."""
    metric_cols = ['evidence_binary_entropy_mean', 'logic_modality_diff', 'final_answer_entropy']
    return evaluate_pair_metrics(df, metric_cols, n_components, test_size)


def run_ablation_study(csv_path: str, model_name: str, n_components: int = 5):
    """
    Run full ablation study.

    Args:
        csv_path: Path to extracted metrics CSV
        model_name: Model name for display
        n_components: Number of GMM components

    Returns:
        DataFrame with ablation results
    """
    df = load_data(csv_path)

    metrics = {
        'H_Evi': 'evidence_binary_entropy_mean',
        'S_Conf': 'logic_modality_diff',
        'H_Ans': 'final_answer_entropy'
    }

    results = []

    # Single metrics
    print("Evaluating individual metrics...")
    for name, col in metrics.items():
        auc = evaluate_single_metric(df, col, n_components)
        if auc:
            results.append({
                'Model': model_name,
                'Metrics': name,
                'AUC': auc
            })
            print(f"  {name}: {auc:.4f}")

    # Pairs
    print("Evaluating metric pairs...")
    metric_list = list(metrics.keys())
    for i in range(len(metric_list)):
        for j in range(i + 1, len(metric_list)):
            pair_name = f"{metric_list[i]} + {metric_list[j]}"
            pair_cols = [metrics[metric_list[i]], metrics[metric_list[j]]]
            auc = evaluate_pair_metrics(df, pair_cols, n_components)
            if auc:
                results.append({
                    'Model': model_name,
                    'Metrics': pair_name,
                    'AUC': auc
                })
                print(f"  {pair_name}: {auc:.4f}")

    # All three
    print("Evaluating all metrics...")
    auc_all = evaluate_all_metrics(df, n_components)
    if auc_all:
        results.append({
            'Model': model_name,
            'Metrics': 'All 3',
            'AUC': auc_all
        })
        print(f"  All 3: {auc_all:.4f}")

    return pd.DataFrame(results)


def compute_synergy_gain(results_df: pd.DataFrame, model_name: str):
    """
    Compute synergy gain: improvement from combining metrics vs best single.

    Args:
        results_df: Results from ablation study
        model_name: Model name

    Returns:
        Dictionary with synergy analysis
    """
    model_results = results_df[results_df['Model'] == model_name]

    # Best single metric
    single_metrics = model_results[~model_results['Metrics'].str.contains('+')]
    best_single_auc = single_metrics['AUC'].max()
    best_single_name = single_metrics.loc[single_metrics['AUC'].idxmax(), 'Metrics']

    # All three metrics
    all_three = model_results[model_results['Metrics'] == 'All 3']
    if len(all_three) > 0:
        all_three_auc = all_three['AUC'].values[0]
    else:
        all_three_auc = best_single_auc

    # Synergy gain
    synergy_gain = all_three_auc - best_single_auc

    # Individual contributions (drop when removing each metric)
    metric_cols = ['evidence_binary_entropy_mean', 'logic_modality_diff', 'final_answer_entropy']
    df = load_data(results_df) if 'evidence_binary_entropy_mean' in pd.DataFrame(results).columns else None

    print(f"\nSynergy Analysis for {model_name}:")
    print(f"  Best single metric: {best_single_name} (AUC={best_single_auc:.4f})")
    print(f"  All three metrics: AUC={all_three_auc:.4f}")
    print(f"  Synergy gain: {synergy_gain:.4f}")

    return {
        'model': model_name,
        'best_single': best_single_name,
        'best_single_auc': best_single_auc,
        'all_three_auc': all_three_auc,
        'synergy_gain': synergy_gain
    }


def main():
    parser = argparse.ArgumentParser(description='Run ablation study for CAD')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to extracted metrics CSV')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name for results')
    parser.add_argument('--n_components', type=int, default=5,
                        help='Number of GMM components')
    parser.add_argument('--output_dir', type=str, default='results/ablation',
                        help='Output directory')

    args = parser.parse_args()

    # Run ablation
    results_df = run_ablation_study(args.input_file, args.model_name, args.n_components)

    # Compute synergy
    synergy = compute_synergy_gain(results_df, args.model_name)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'{args.model_name}_ablation.csv')
    results_df.to_csv(output_file, index=False)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
