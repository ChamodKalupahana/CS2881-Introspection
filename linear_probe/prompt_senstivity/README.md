# Prompt Sensitivity Analysis

This folder contains tools for analyzing how sensitive the introspection probe is to different prompt formats, and for calculating a **Mass-Mean direction** that specifically distinguishes between:
1.  **Positive Case**: The model detects a truly injected concept vector.
2.  **Negative Case (Calibration)**: The model "parrots" detection because the concept was mentioned in the text prompt (but not injected into activations).

## Key Components

- **`save_activations_prompts.py`**: Runs a battery of separate textual prompts (positive and negative) to collect activation data.
  - Saves `last_token`, `prompt_last_token`, and `all_generation` activations.
  - Organizes output into `saved_activations/prompt_activations_<timestamp>/` with subfolders for `positive/` and `negative/` categories.

- **`save_activations_prompts_unified.py`**: An updated alternative script that uses a singular set of `UNIFIED_PROMPTS` for both positive (injected) and negative (calibration) conditions. Evaluates model sensitivity based on *how* the concept is presented (weights vs. text) rather than fundamentally different prompts.

- **`judge_all_prompts.py`**: Centralized LLM judge infrastructure.
  - Implements a 3-stage pipeline: **Coherence** → **Affirmative Detection** → **Category Classification**.
  - Used by all scripts in this folder for consistent evaluation.

- **`compute_mass_mean_vector_prompts.py`**: Calculates the steering vector pointing from "Calibration Correct" to "Injected Correct".
  - This vector represents the specific internal state of "real" introspection.
  - Generates projection plots and logit lens analysis.

- **`compute_PCA.py`**: Analyzes the dimensionality of the introspection signal across the network.
  - Automatically matches positive and negative activations to compute a pure "introspection delta".
  - Computes Principal Component Analysis (PCA) on these deltas for all layers (16→31).
  - Outputs a 2D heatmap matrix showing the explained variance ratio for the top *k* components per layer.

- **`test_probe_dir_casual_prompts.py`**: Tests the learned probe by steering the model's activations during generation.
  - Sweeps over `alpha` (scaling factor) to see if amplifying/suppressing the probe direction changes the model's detection behavior.

---

## Helpful Commands

### 1. Collect Activation Data
```bash
python save_activations_prompts.py --layer 16 --coeff 8.0 --datasets simple_data

# Or use the unified prompts version:
python save_activations_prompts_unified.py --layer 16 --coeff 8.0 --datasets simple_data
```

### 2. Compute the Probe Vector
```bash
# Using the last token of the full response
python compute_mass_mean_vector_prompts.py --layer 19 --token_type last_token

# Using the last token of the input prompt
python compute_mass_mean_vector_prompts.py --layer 19 --token_type prompt_last_token
```

### 3. Analyze Dimensionality via PCA
```bash
# Computes the variance explained by the top 8 PCA components across all recorded layers and plots a heatmap
python compute_PCA.py --max_components 8
```

### 4. Test the Probe (Steering)
```bash
python test_probe_dir_casual_prompts.py \
    --layers 16 --coeffs 8.0 \
    --datasets test_data \
    --alphas 0 2 5 10 \
    --probe_layer 19
```

### 5. Full Pipeline (via main.sh)
```bash
bash main.sh
```

## Data Structure
- `probe_vectors/`: Saved `.pt` files for the calculated direction vectors.
- `plots/`: (Centralized) Visualizations of class separation and logit lens results.
- `saved_activations/`: Raw activation data organized by run and classification category.
