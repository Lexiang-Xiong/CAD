"""
Prompt templates and utility functions for various VLM models.

This module provides:
- Uncertainty word lists for H_Evi calculation
- Model-specific Chain-of-Thought (CoT) prompt templates
- Helper functions for token ID extraction
"""

from typing import List, Set

# Uncertainty-related words for perceptual instability detection
UNCERTAINTY_WORDS = [
    "probably", "likely", "possibly", "might", "may", "seems", "appears",
    "perhaps", "suggests", "could", "believe", "guess", "assume", "unlikely",
    "not sure", "could be", "maybe", "perhaps", "I think", "I believe",
    "it's hard to say", "difficult to determine", "uncertain"
]


def get_uncertainty_token_ids(tokenizer) -> List[int]:
    """
    Generate tokenizer-compatible token IDs for uncertainty words.

    Args:
        tokenizer: HuggingFace tokenizer instance

    Returns:
        List of unique token IDs corresponding to uncertainty words
    """
    token_ids = set()
    for word in UNCERTAINTY_WORDS:
        # Handle both with and without leading space
        for prefix in [" ", ""]:
            text = prefix + word
            ids = tokenizer.encode(text, add_special_tokens=False)
            token_ids.update(ids)
    return list(token_ids)


# ============================================================================
# Model-Specific Prompt Templates
# ============================================================================

# LLaVA Prompt Template
LLAVA_PROMPT = """[INST] <image>
{question}. First, provide a brief explanation of what you see in the image. Then, conclude with 'Therefore, the final answer is Yes.' or 'Therefore, the final answer is No.' [/INST]"""


# Qwen2-VL Prompt Template
QWEN2_PROMPT = """Question: {question}
You must strictly follow the format below:
1. Describe what you see in the image in detail.
2. End your response with either 'Therefore, the final answer is Yes.' or 'Therefore, the final answer is No.'

"""


# Idefics2 Prompt Template
IDEFICS2_PROMPT = """User: <image>
You are an expert image analyst. Follow this output format strictly:

First, provide a brief explanation of what you see in the image. Then, conclude with 'Therefore, the final answer is Yes.' or 'Therefore, the final answer is No.'
Question: {question}

Assistant:
"""


# DeepSeek-VL2 Prompt Template
DEEPSEEK_PROMPT = """User: <image>
Please analyze this image carefully and answer the following question.
{question}

Please provide your reasoning and end with 'Therefore, the final answer is Yes.' or 'Therefore, the final answer is No.'

Assistant:"""


def get_prompt_template(model_name: str) -> str:
    """
    Get the appropriate prompt template for a specific model.

    Args:
        model_name: Name of the VLM model

    Returns:
        Prompt template string with {question} placeholder
    """
    model_name_lower = model_name.lower()

    if "llava" in model_name_lower:
        return LLAVA_PROMPT
    elif "qwen" in model_name_lower:
        return QWEN2_PROMPT
    elif "idefics" in model_name_lower:
        return IDEFICS2_PROMPT
    elif "deepseek" in model_name_lower:
        return DEEPSEEK_PROMPT
    else:
        # Default to LLaVA format
        return LLAVA_PROMPT


def format_prompt(template: str, question: str) -> str:
    """
    Format a prompt template with a specific question.

    Args:
        template: Prompt template with {question} placeholder
        question: The question to insert

    Returns:
        Formatted prompt string
    """
    return template.format(question=question)


# ============================================================================
# Answer Parsing Utilities
# ============================================================================

def extract_answer_from_response(response: str) -> str:
    """
    Extract final answer (Yes/No) from model response.

    Args:
        response: Full model response text

    Returns:
        'Yes', 'No', or 'Unknown'
    """
    response_lower = response.lower()

    # Look for the specific answer format
    if "therefore, the final answer is yes" in response_lower:
        return "Yes"
    elif "therefore, the final answer is no" in response_lower:
        return "No"

    # Fallback: look for any yes/no in the response
    if "yes" in response_lower and "no" not in response_lower.split("yes")[0]:
        return "Yes"
    elif "no" in response_lower and "yes" not in response_lower.split("no")[0]:
        return "No"

    return "Unknown"


def get_target_token_ids(tokenizer, answer: str = "Yes") -> List[int]:
    """
    Get token IDs for target answer words.

    Args:
        tokenizer: HuggingFace tokenizer
        answer: Answer string ("Yes" or "No")

    Returns:
        List of token IDs for the answer
    """
    # Get both with and without capitalization
    ids = set()
    ids.update(tokenizer.encode(answer, add_special_tokens=False))
    ids.update(tokenizer.encode(answer.lower(), add_special_tokens=False))

    return list(ids)
