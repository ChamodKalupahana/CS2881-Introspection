# Introspection: Understanding and Steering LLM Internal States

![Project Banner](https://img.shields.io/badge/Research-AI_Safety_&_Interpretability-blueviolet?style=for-the-badge)
![Python Version](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Framework](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

## 🚀 Overview
This repository contains the codebase for the **Introspection** project, a research initiative focused on deciphering and manipulating the internal representations of Large Language Models (LLMs). We employ a variety of techniques—from linear probing and PCA to mechanistic interpretability via activation patching—to understand how models represent concepts and how these representations can be steered.

## 🔬 Core Methodologies

### 1. Concept Steering & Vector Injection
We compute **Concept Vectors** (e.g., "Truthfulness", "Success", "Angry") by analyzing the difference in activations between contrastive datasets. These vectors can then be injected back into the model during inference to influence its behavior.
- **Key Scripts**: `original_paper/save_vectors.py`, `original_paper/inject_concept_vector.py`

### 2. Linear Probing & Activation Analysis
We train **Linear Probes** on internal activations to detect specific properties at different layers. This allows us to "read the model's mind" and determine if it "knows" certain concepts even when they aren't explicitly output.
- **Key Scripts**: `linear_probe/train_probe.py`, `linear_probe/validate_across_prompt.py`

### 3. Mechanistic Interpretability
We use **Activation Patching** and **Direct Logit Attribution (DLA)** to pinpoint the specific attention heads and MLP layers responsible for model behavior.
- **Key Scripts**: `acitvation_patching/patch_heads.py`, `DLA/DLA_towards_probe_dir.py`

### 4. Dimensionality Reduction (PCA)
By performing **Principal Component Analysis** on activations, we visualize the "shape" of concepts in the high-dimensional representation space.
- **Key Scripts**: `PCA/compute_PCA.py`, `PCA/validate_PCA.py`

---

## 📁 Project Structure

```text
.
├── original_paper/       # Core implementation and main experiment entry points
├── linear_probe/         # Training and evaluating linear classifiers on activations
├── acitvation_patching/  # Pinpointing behavior to specific model components
├── DLA/                  # Direct Logit Attribution analysis
├── PCA/                  # Dimensionality reduction and visualization
├── model_utils/          # Shared utilities for logging, judges, and injection
├── success_programs/     # Analysis of "success directions" and introspection
├── dataset/              # JSON datasets for training and evaluation
├── plots/                # Generated visualizations and analysis plots
├── success_results/      # Automatically generated experiment logs and results
└── README.md             # You are here
```

---

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.10+
- PyTorch (with CUDA/MPS support)
- HuggingFace Transformers & Accelerate

### 2. Installation
```bash
git clone https://github.com/ChamodKalupahana/CS2881-Introspection.git
cd CS2881-Introspection
pip install -r requirements.txt
```

### 3. Configuration
Set up your environment variables (especially for OpenAI-based judges):
```bash
cp .env.example .env
# Edit .env with your keys
```

---

## 📊 Example Usage

### Compute and Save Concept Vectors
```bash
python original_paper/save_vectors.py \
    --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --datasets simple_data complex_data \
    --layer_range 0 16 31 \
    --save_dir "saved_vectors/llama"
```

### Train a Linear Probe
```bash
python linear_probe/train_probe.py --dataset simple_data --layer 16
```

### Run Main Introspection Experiment
```bash
python original_paper/main.py --experiment mcq --model llama-3.1-8b
```

---

### 📊 Paper Figures & Source Code

| Figure # | Folder / Topic | Description & Source Context |
| :--- | :--- | :--- |
| **Fig. 1-2** | `dataset/` | Schematics of concept vector construction (simple vs. complex). |
| **Fig. 4-5** | `acitvation_patching/` | Activation patching heatmaps. Generated via `patch_heads.py`. |
| **Fig. 6** | `DLA/` | Direct Logit Attribution analysis for concept components. |
| **Fig. 7-8** | `PCA/` | PCA projections of residual stream activations. See `compute_PCA.py`. |
| **Fig. 9** | `plots/linear_probe/` | Sweep of layer and coefficient for concept steering. |
| **Fig. 10** | `linear_probe/` | Separation quality analysis for difference-in-means vectors. |
| **Fig. 11-13** | `linear_probe/` | Dot product projections and 2D Activation Subspace experiments. |
| **Fig. 14** | `plots/` | Strength detection layer sweep results. |
| **Fig. 15-16** | `PCA/` | Cosine similarity heatmaps for PCA components (Appreciation/Refusal). |
| **Fig. 17** | `model_utils/` | Ablation of the refusal direction. See `model_utils/injection.py`. |
| **Fig. 18-19** | `linear_probe/` | Appendix: Detection metrics and "not_detected" separation quality. |
| **Fig. 20** | `model_utils/` | Evaluation of LLM judges (Qwen, Llama-3.2) in `llm_judges.py`. |
| **Fig. 21-22** | `linear_probe/` | Steering results along introspection directions and dot product distributions. |
| **Fig. 23-24** | `plots/linear_probe/` | Logit lens results for specific probe vectors (CAL/ND). |

---

## 🏗️ Future Directions
- **Endogenous Steering Resistance (ESR)**: Investigating how models resist external concept steering.

---

## ⚠️ Notes
- **.pt Files**: Saved PyTorch tensors are ignored by git to avoid bloating the repository.
- **Environment**: Ensure your python environment has access to the project root. Scripts handle `sys.path` modification, but running from the root directory is recommended.