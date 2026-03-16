"""
Core metrics for information-theoretic probe calculations.

This module provides pure mathematical implementations for computing:
- H_Evi: Perceptual Instability (Evidence Binary Entropy)
- S_Conf: Inferential Conflict (Modality Difference Rate / CPMI)
- H_Ans: Decisional Ambiguity (Standard Shannon Entropy)

These metrics are model-agnostic and can be used with any VLM.
"""

import torch
import numpy as np
from typing import List

EPSILON = 1e-9


@torch.inference_mode()
def calculate_standard_entropy(probs: torch.Tensor) -> float:
    """
    Calculate standard Shannon entropy (H_Ans: Decisional Ambiguity).

    Measures the uncertainty in the final answer distribution.
    Higher entropy indicates the model is less confident about its answer.

    Args:
        probs: Probability tensor over token vocabulary

    Returns:
        Entropy value in bits (clamped to non-negative)
    """
    probs_non_zero = probs[probs > 0]
    if probs_non_zero.numel() == 0:
        return 0.0
    entropy = -torch.sum(probs_non_zero * torch.log2(probs_non_zero)).item()
    return max(0.0, entropy)


@torch.inference_mode()
def calculate_binary_entropy(probs: torch.Tensor, uncertain_token_ids: List[int]) -> float:
    """
    Calculate uncertainty binary entropy (H_Evi: Perceptual Instability).

    Measures the instability of evidence tokens by computing the entropy
    over the probability mass assigned to uncertainty-related tokens.

    Args:
        probs: Probability tensor over token vocabulary
        uncertain_token_ids: List of token IDs corresponding to uncertainty words

    Returns:
        Binary entropy value in bits
    """
    valid_ids = [tid for tid in uncertain_token_ids if tid < len(probs)]
    if not valid_ids:
        return 0.0

    p_unc = torch.sum(probs[torch.tensor(valid_ids, dtype=torch.long)]).item()
    if p_unc < EPSILON or (1.0 - p_unc) < EPSILON:
        return 0.0

    return -(p_unc * np.log2(p_unc) + (1.0 - p_unc) * np.log2(1.0 - p_unc))


@torch.inference_mode()
def calculate_modality_diff_rate(prob_vision: torch.Tensor, prob_text: torch.Tensor, token_id: int) -> float:
    """
    Calculate modality difference rate (S_Conf: Inferential Conflict / CPMI).

    Measures the conflict between vision-conditioned and text-only predictions.
    This captures whether the model relies too heavily on either modality.

    Args:
        prob_vision: Probability distribution when conditioned on image
        prob_text: Probability distribution from text-only replay
        token_id: Target token ID for comparison

    Returns:
        Log probability difference (CPMI)
    """
    p_v = prob_vision[token_id].item() if token_id < len(prob_vision) else 0
    p_t = prob_text[token_id].item() if token_id < len(prob_text) else 0

    # Log difference measures conditional mutual information gain
    log_diff = np.log(p_v + EPSILON) - np.log(p_t + EPSILON)
    return log_diff


def compute_all_metrics(
    vision_dist: torch.Tensor,
    text_dist: torch.Tensor,
    final_answer_dist: torch.Tensor,
    uncertain_token_ids: List[int],
    target_token_id: int
) -> dict:
    """
    Compute all three information-theoretic metrics for a single sample.

    Args:
        vision_dist: Token probabilities with image conditioning
        text_dist: Token probabilities from text-only replay
        final_answer_dist: Final answer token distribution
        uncertain_token_ids: Token IDs for uncertainty words
        target_token_id: Token ID for "Yes" or "No" answer

    Returns:
        Dictionary containing H_Evi, S_Conf, and H_Ans values
    """
    h_evi = calculate_binary_entropy(vision_dist, uncertain_token_ids)
    s_conf = calculate_modality_diff_rate(vision_dist, text_dist, target_token_id)
    h_ans = calculate_standard_entropy(final_answer_dist)

    return {
        'H_Evi': h_evi,
        'S_Conf': s_conf,
        'H_Ans': h_ans
    }
