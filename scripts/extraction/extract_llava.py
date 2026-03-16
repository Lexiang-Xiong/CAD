"""
Feature extraction script for LLaVA model.

This script performs:
1. Forward pass with images to extract vision-conditioned distributions
2. Text-only replay (non-autoregressive intervention) to extract text distributions
3. Calculation of cognitive metrics (H_Evi, S_Conf, H_Ans)

Usage:
    python scripts/extraction/extract_llava.py \
        --dataset lmms-lab/POPE \
        --split test \
        --output_dir results/llava
"""

import os
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.metrics.core_metrics import calculate_binary_entropy, calculate_modality_diff_rate, calculate_standard_entropy
from src.utils.prompt_utils import get_uncertainty_token_ids, get_prompt_template, format_prompt, extract_answer_from_response


def extract_vision_distribution(model, processor, prompt, image, device):
    """Extract token distributions with image conditioning."""
    inputs = processor(prompt, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]  # Shape: (seq_len, vocab_size)
        probs = torch.softmax(logits, dim=-1)

    return probs


def extract_text_distribution(model, processor, prompt, device):
    """Extract token distributions without image (text-only replay)."""
    # Create a dummy image (black pixel) for text-only input
    from PIL import Image

    # Get image size from processor
    dummy_image = Image.new('RGB', (336, 336), color='black')

    inputs = processor(prompt, images=dummy_image, return_tensors="pt").to(device)

    # Mask out image tokens by setting their positions to a special token
    # This ensures we only get text-conditioned predictions
    image_token_id = processor.tokenizer.unk_token_id

    # Forward pass with masked vision
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)

    return probs


def process_sample(model, processor, uncertain_token_ids, prompt_template, sample, device):
    """Process a single sample and compute cognitive metrics."""
    try:
        question = sample['question']
        image = sample['image']
        ground_truth = sample.get('ground_truth', sample.get('label', None))

        # Format prompt
        prompt = format_prompt(prompt_template, question)

        # Extract distributions
        vision_dist = extract_vision_distribution(model, processor, prompt, image, device)
        text_dist = extract_text_distribution(model, processor, prompt, device)

        # Get final answer distribution (last token position)
        # Find the position after "Therefore, the final answer is"
        answer_prefix = "Therefore, the final answer is"
        answer_ids = processor.encode(answer_prefix, add_special_tokens=False)

        # Find the token position after the answer prefix
        # We use a simpler approach: get the distribution at the last token
        final_answer_dist = vision_dist[-1]

        # Get target token IDs for "Yes" and "No"
        yes_ids = processor.encode("Yes", add_special_tokens=False)
        no_ids = processor.encode("No", add_special_tokens=False)
        target_token_id = yes_ids[0] if yes_ids else 0

        # Calculate metrics
        h_evi = calculate_binary_entropy(vision_dist, uncertain_token_ids)
        s_conf = calculate_modality_diff_rate(vision_dist, text_dist, target_token_id)
        h_ans = calculate_standard_entropy(final_answer_dist)

        # Extract model answer from text
        full_response = processor.decode(vision_dist.argmax(dim=-1), skip_special_tokens=True)
        model_answer = extract_answer_from_response(full_response)

        # Determine if hallucination (compare with ground truth)
        is_hallucination = None
        if ground_truth is not None:
            # Handle different label formats
            if isinstance(ground_truth, str):
                is_hallucination = 1 if model_answer.lower() != ground_truth.lower() else 0
            elif isinstance(ground_truth, int):
                # Assume 1 = yes, 0 = no
                true_answer = "Yes" if ground_truth == 1 else "No"
                is_hallucination = 1 if model_answer != true_answer else 0

        return {
            'question': question,
            'ground_truth': ground_truth,
            'model_answer': model_answer,
            'is_hallucination': is_hallucination,
            'evidence_binary_entropy_mean': h_evi,
            'logic_modality_diff': s_conf,
            'final_answer_entropy': h_ans,
            'status': 'success'
        }

    except Exception as e:
        return {
            'question': sample.get('question', ''),
            'status': 'failed',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Extract cognitive metrics from LLaVA')
    parser.add_argument('--model_id', type=str, default='llava-hf/llava-v1.6-mistral-7b-hf',
                        help='HuggingFace model ID')
    parser.add_argument('--dataset', type=str, default='lmms-lab/POPE',
                        help='Dataset name on HuggingFace')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to use')
    parser.add_argument('--output_dir', type=str, default='results/llava',
                        help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (currently only 1 supported)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and processor
    print(f"Loading model: {args.model_id}")
    processor = LlavaNextProcessor.from_pretrained(args.model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )
    model.eval()

    # Get uncertainty token IDs
    uncertain_token_ids = get_uncertainty_token_ids(processor.tokenizer)

    # Get prompt template
    prompt_template = get_prompt_template(args.model_id)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split=args.split)

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Processing {len(dataset)} samples...")

    # Process samples
    results = []
    for sample in tqdm(dataset, desc="Extracting cognitive metrics"):
        result = process_sample(
            model, processor, uncertain_token_ids, prompt_template, sample, device
        )
        results.append(result)

    # Save results
    df = pd.DataFrame(results)
    output_file = os.path.join(args.output_dir, 'hallucination_metrics_full.csv')
    df.to_csv(output_file, index=False)

    # Print summary
    success_count = len(df[df['status'] == 'success'])
    print(f"\nExtraction complete!")
    print(f"  Total samples: {len(df)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {len(df) - success_count}")
    print(f"  Results saved to: {output_file}")


if __name__ == '__main__':
    main()
