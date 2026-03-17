<div align="center">
<h1><a href="https://arxiv.org/abs/2603.15557" target="_blank">Anatomy of a Lie: A Multi-Stage Diagnostic Framework for Tracing Hallucinations in Vision-Language Models</a></h1>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/arXiv-2603.15557-b31b1b.svg)](https://arxiv.org/abs/2603.15557)


</div>

<div align="center">
Lexiang Xiong*&emsp;Qi Li*&emsp;Jingwen Ye&emsp;Xinchao Wang<sup>&dagger;</sup>
</div>
<div align="center">
    <a href="https://sites.google.com/view/xml-nus/people?authuser=0" target="_blank">xML-Lab</a>, National University of Singapore&emsp;
    <sup>&dagger;</sup>corresponding author;
    <sup>*;</sup>equal contribition;
    
</div>
</div>
</div>


## Overview

Vision-Language Models (VLMs) frequently hallucinate. Current evaluation methods treat these errors as static, monolithic failures. In this work, we propose a paradigm shift: viewing hallucination as a dynamic pathology within a model's computational cognition.

We introduce **Cognitive Anomaly Detection (CAD)**, a framework that projects a VLM's generative process onto an interpretable, low-dimensional **Cognitive State Space** using three novel information-theoretic probes:

1. **Perceptual Instability ($H_{Evi}$)** - Measures uncertainty in evidence tokens
2. **Logical-Causal Failure ($S_{Conf}$)** - Captures inferential conflict between vision and text
3. **Decisional Ambiguity ($H_{Ans}$)** - Measures uncertainty in final answer

By leveraging **geometric-information duality**, CAD diagnoses hallucinations as geometric anomalies with high information-theoretic surprisal, requiring **only a single generation pass** and **weak supervision** (no token-level hallucination labels needed).

## Key Features

- **Mechanistic Diagnosis**: Uncovers distinct failure modes (e.g., *Computational Cognitive Dissonance*, *Transparent Struggles*, *Entangled States*)
- **High Efficiency**: Single-pass generation + lightweight non-autoregressive replay
- **Weak Supervision**: Calibrates on a small set of ground-truth answers (resilient to up to 30% calibration contamination)
- **State-of-the-Art**: Achieves superior AUC across POPE, MME, and MS-COCO on models like Idefics2, Llava-v1.6, Qwen2-VL, and DeepSeek-VL2

## Installation

```bash
# Clone the repository
git clone https://github.com/Lexiang-Xiong/CAD
cd Anatomy-of-a-Lie

# Create conda environment
conda create -n cad-vlm python=3.10 -y
conda activate cad-vlm

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r requirements.txt

# For Qwen2-VL support
pip install qwen-vl-utils

# For DeepSeek-VL2 support
pip install deepseek_vl2
```

### Environment Variables

Set your HuggingFace token for accessing gated models:

```bash
export HF_TOKEN="your_huggingface_token"
```

## Quick Start

### 1. Extract Cognitive Trajectories

Run the extraction scripts to generate responses, perform text-only replay, and calculate the core metrics ($H_{Evi}$, $S_{Conf}$, $H_{Ans}$).

```bash
# Idefics2
python scripts/extraction/extract_idefics2.py \
    --dataset lmms-lab/POPE \
    --output_dir results/idefics2

# LLaVA
python scripts/extraction/extract_llava.py \
    --dataset lmms-lab/POPE \
    --output_dir results/llava

# Qwen2-VL
python scripts/extraction/extract_qwen2.py \
    --dataset lmms-lab/POPE \
    --output_dir results/qwen2

# DeepSeek-VL2
python scripts/extraction/extract_deepseek.py \
    --dataset lmms-lab/POPE \
    --output_dir results/deepseek
```

### 2. Fit CAD Detector and Evaluate

Calibrate the GMM on the nominal state manifold and predict hallucinations on the test set.

```bash
# Basic evaluation
python scripts/evaluation/run_cad_eval.py \
    --input_file results/idefics2/hallucination_metrics_full.csv \
    --model_name Idefics2 \
    --n_components 7

# Automatic K selection using BIC
python scripts/evaluation/run_cad_eval.py \
    --input_file results/idefics2/hallucination_metrics_full.csv \
    --model_name Idefics2 \
    --auto_k
```

### 3. Run Ablation Study

Evaluate the contribution of each metric component.

```bash
python scripts/evaluation/run_ablation.py \
    --input_file results/idefics2/hallucination_metrics_full.csv \
    --model_name Idefics2
```

## Reproducing Paper Figures

We provide visualization scripts used in the paper.

```bash
# Figure 2: ROC Panels (Linear & Log-Log)
python visualizations/plot_roc.py

# Figure 3: Cognitive Manifold Fingerprints (2x2 KDE)
python visualizations/plot_manifold.py

# Figure 4: Ablation Study & Synergy Gain
python visualizations/plot_ablation.py

# Figure 6: Robustness to Calibration Contamination
python visualizations/plot_robustness.py
```

## Repository Structure

```
Anatomy-of-a-Lie/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment (optional)
│
├── src/                        # Core algorithm modules
│   ├── metrics/
│   │   └── core_metrics.py     # Information-theoretic probe implementations
│   ├── detector/
│   │   └── cad_gmm.py          # GMM-based cognitive anomaly detector
│   └── utils/
│       └── prompt_utils.py     # Prompt templates and utilities
│
├── scripts/                    # Experiment execution scripts
│   ├── extraction/             # Feature extraction scripts
│   │   ├── extract_llava.py
│   │   ├── extract_idefics2.py
│   │   ├── extract_qwen2.py
│   │   └── extract_deepseek.py
│   └── evaluation/             # Evaluation scripts
│       ├── run_cad_eval.py
│       └── run_ablation.py
│
├── visualizations/             # Plotting scripts
│   ├── plot_roc.py
│   ├── plot_manifold.py
│   ├── plot_ablation.py
│   └── plot_robustness.py
│
└── data/                       # Data directory
    └── README.md               # Instructions for dataset preparation
```

## Supported Models

- LLaVA-v1.6-Mistral-7B
- Idefics2-8B
- Qwen2-VL-7B-Instruct
- DeepSeek-VL2-Tiny/Small

## Supported Datasets

- [POPE](https://github.com/AoiDragon/POPE)
- [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)
- MS-COCO

## Citation

If you find our work or this codebase helpful, please consider citing our paper:

```bibtex
@article{anonymous2026anatomy,
  title={Anatomy of a Lie: A Multi-Stage Diagnostic Framework for Tracing Hallucinations in Vision-Language Models},
  author={Anonymous},
  year={2026}
}
```

## Acknowledgements

We thank the open-source VLM community, particularly the maintainers of [POPE](https://github.com/RUCAIBox/POPE), [Llava](https://github.com/haotian-liu/LLaVA), [Idefics2](https://huggingface.co/blog/idefics2), [Qwen-VL](https://github.com/QwenLM/Qwen-VL), and [DeepSeek-VL](https://github.com/deepseek-ai/DeepSeek-VL2) for their invaluable resources.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
