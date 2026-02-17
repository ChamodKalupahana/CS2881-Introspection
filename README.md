# Introspection Project

This project explores concept vectors, introspection, and activation patching in large language models. It provides tools to compute concept vectors, inject them during inference, and analyze the results using various metrics and probes.

## Installation

### 1. Clone the Repository
```bash
git clone <repository_url>
cd CS2881-Introspection
```

### 2. Set up Environment
Create a virtual environment (conda or venv) and install dependencies:
```bash
# Using pip
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Using conda
conda create -n introspection python=3.10
conda activate introspection
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy the example environment file and add your OpenAI API key:
```bash
cp .env.example .env
```
Edit `.env` and set `OPENAI_API_KEY`:
```
OPENAI_API_KEY="your-api-key-here"
```

## Project Structure

- **`original_paper/`**: Core implementation of the project.
  - `compute_concept_vector_utils.py`: Utility functions for computing concept vectors.
  - `inject_concept_vector.py`: Logic for injecting vectors during inference.
  - `save_vectors.py`: Script to sweep layers and compute/save concept vectors.
  - `main.py`: Main entry point for running experiments (e.g., measuring introspection capabilities).
- **`success_programs/`**: Scripts for verifying success directions and analyzing model behavior.
- **`acitvation_patching/`**: Scripts for activation patching experiments (attention heads, MLP layers).
- **`DLA/`**: Direct Logit Attribution analysis scripts.
- **`dataset/`**: JSON datasets used for computing concept vectors (e.g., `simple_data.json`).
- **`success_results/`**: Directory where experiment results are saved (created automatically).

## Usage

### 1. Compute and Save Concept Vectors
To compute concept vectors across layers and datasets:
```bash
python original_paper/save_vectors.py \
    --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --datasets simple_data complex_data \
    --layer_range 0 1 2 3 ... 31 \
    --save_dir "saved_vectors/llama"
```
Arguments:
- `--model`: HuggingFace model ID (default: `meta-llama/Meta-Llama-3.1-8B-Instruct`).
- `--datasets`: List of datasets to process (e.g., `simple_data`, `complex_data`).
- `--layer_range`: Layers to sweep (default: 0-31).
- `--save_dir`: Output directory for saved `.pt` vectors.

### 2. Run Main Experiments
To run the main introspection experiments:
```bash
python original_paper/main.py
```
This script runs various experiment types (e.g., multiple choice, generative distinguishability) defined in `main.py`.

### 3. Run Verification/Analysis Scripts
Example: Running activation patching on attention heads:
```bash
python acitvation_patching/patch_heads.py --inject_layer 16 --coeff 10.0
```

## Outputs

Each run typically creates a timestamped directory in `success_results/` (e.g., `success_results/run_MM_DD_YY_HH_MM/`).
Logs, saved vectors (if applicable), and analysis results are stored there.

## Notes

- **.pt Files**: Saved PyTorch tensors (`.pt` files) are ignored by git to avoid bloating the repository.
- **Environment**: Ensure your python environment has access to the project root (the scripts handle `sys.path` modification, but running from the root `CS2881-Introspection` directory is recommended).