"""
CAD Evaluation Script: Compute AUC and evaluate detection performance.

This script:
1. Loads extracted cognitive metrics from CSV
2. Fits the GMM-based Cognitive Anomaly Detector on nominal samples
3. Computes AUC-ROC scores for hallucination detection
4. Optionally runs hyperparameter search for optimal K

Usage:
    python scripts/evaluation/run_cad_eval.py \
        --input_file results/idefics2/hallucination_metrics_full.csv \
        --model_name Idefics2 \
        --n_components 7

    # With automatic K selection using BIC:
    python scripts/evaluation/run_cad_eval.py \
        --input_file results/idefics2/hallucination_metrics_full.csv \
        --model_name Idefics2 \
        --auto_k
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.detector.cad_gmm import CognitiveAnomalyDetector, select_optimal_k


def load_and_preprocess_data(csv_path: str):
    """
    Load extracted metrics and preprocess for CAD evaluation.

    Args:
        csv_path: Path to the extracted metrics CSV file

    Returns:
        X: Feature matrix (N, 3) with [H_Evi, S_Conf, H_Ans]
        y: Binary labels (1 = hallucination, 0 = correct)
        df: Original DataFrame
    """
    df = pd.read_csv(csv_path)

    # Drop failed samples
    df = df[df['status'] == 'success']

    # Drop rows with missing values in required columns
    required_cols = ['evidence_binary_entropy_mean', 'logic_modality_diff', 'final_answer_entropy']
    df = df.dropna(subset=required_cols + ['is_hallucination'])

    # Extract features and labels
    X = df[required_cols].values
    y = df['is_hallucination'].values.astype(int)

    print(f"Loaded {len(X)} samples from {csv_path}")
    print(f"  Hallucination rate: {y.mean():.2%}")

    return X, y, df


def evaluate_cad(X, y, n_components: int, test_size: float = 0.3, random_state: int = 42):
    """
    Evaluate CAD detector with train/test split.

    Args:
        X: Feature matrix
        y: Labels
        n_components: Number of GMM components
        test_size: Fraction for test split
        random_state: Random seed

    Returns:
        Dictionary with evaluation results
    """
    # Split data
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Extract nominal samples (correct answers) from pool for calibration
    X_nominal = X_pool[y_pool == 0]

    print(f"Calibration set size: {len(X_nominal)} (correct answers only)")

    # Fit detector
    detector = CognitiveAnomalyDetector(n_components=n_components, random_state=random_state)
    detector.fit(X_nominal)

    # Predict on test set
    surprisal_scores = detector.predict_surprisal(X_test)

    # Calculate AUC
    auc = roc_auc_score(y_test, surprisal_scores)

    # Get ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, surprisal_scores)

    return {
        'auc': auc,
        'fpr': fpr,
        'tpr': tpr,
        'n_components': n_components,
        'test_size': len(X_test),
        'calibration_size': len(X_nominal)
    }


def run_k_search(X, y, k_range: range):
    """
    Run hyperparameter search for optimal K using BIC.

    Args:
        X: Feature matrix
        y: Labels
        k_range: Range of K values to try

    Returns:
        Dictionary with search results
    """
    # Use only correct samples for calibration
    X_nominal = X[y == 0]

    print(f"Running BIC-based K selection with range {list(k_range)}...")

    result = select_optimal_k(X_nominal, k_range)

    print(f"Optimal K: {result['k']}")
    print(f"BIC scores: {result['bic_scores']}")

    return result


def main():
    parser = argparse.ArgumentParser(description='Evaluate CAD hallucination detector')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to extracted metrics CSV')
    parser.add_argument('--model_name', type=str, default='VLM',
                        help='Model name for display')
    parser.add_argument('--n_components', type=int, default=5,
                        help='Number of GMM components (K)')
    parser.add_argument('--auto_k', action='store_true',
                        help='Automatically select K using BIC')
    parser.add_argument('--k_range', type=int, nargs=2, default=[2, 12],
                        help='Range for K selection (min max)')
    parser.add_argument('--test_size', type=float, default=0.3,
                        help='Test set fraction')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                        help='Output directory for results')

    args = parser.parse_args()

    # Load data
    X, y, df = load_and_preprocess_data(args.input_file)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Auto K selection or use provided K
    if args.auto_k:
        k_result = run_k_search(X, y, range(args.k_range[0], args.k_range[1]))
        optimal_k = k_result['k']
        print(f"\nUsing optimal K = {optimal_k}")
    else:
        optimal_k = args.n_components

    # Evaluate
    result = evaluate_cad(
        X, y,
        n_components=optimal_k,
        test_size=args.test_size
    )

    print(f"\n{'='*50}")
    print(f"CAD Evaluation Results ({args.model_name})")
    print(f"{'='*50}")
    print(f"  K (components): {result['n_components']}")
    print(f"  AUC-ROC: {result['auc']:.4f}")
    print(f"  Calibration samples: {result['calibration_size']}")
    print(f"  Test samples: {result['test_size']}")

    # Save results
    output_file = os.path.join(args.output_dir, f'{args.model_name}_cad_results.csv')
    results_df = pd.DataFrame([{
        'model': args.model_name,
        'k_components': result['n_components'],
        'auc': result['auc'],
        'calibration_samples': result['calibration_size'],
        'test_samples': result['test_size']
    }])
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
