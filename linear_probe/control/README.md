# Control Activations

This directory contains logic to compute, extract, and analyze activation states for complex concept mapping across specific layers of a Causal Language Model. 

## `save_activations_and_concept.py`

This script computes the component activation distributions inside a model given a target concept and dataset. It dynamically aggregates activations directly from the underlying layers (all layers strictly greater than the user-specified layer threshold) and runs them through a single cohesive forward pass for maximum performance. 

By obtaining these exact continuous vector representations (the positive and negative examples per text), it sets up your pipeline for mass-mass vector projection, Principal Component Analysis (PCA), and Cohen’s *d* evaluation.

### Usage Example
You can pass arguments to map against the model dynamically:
```bash
cd linear_probe/control/
python save_activations_and_concept.py --dataset_name "complex_data" --concept_name "appreciation" --min_layer_to_save 16
```

### Command-line Arguments

- `--dataset_name` (str, **Required**): Target dataset mapping (e.g. `test_data` or `complex_data`).
- `--concept_name` (str, **Required**): Name of the overarching conceptual token to probe.
- `--model` (str): Model name or local absolute path. By default, it loads `meta-llama/Meta-Llama-3.1-8B-Instruct`.
- `--min_layer_to_save` (int): Threshold index point. The script will map and capture activations for every hidden layer *above* this integer. Defaults to `16`.

## `test_probe_dir_to_simple_baseline.py`

This script compares your trained probe vectors (both Mass-Mean Vectors and PCA components) against a simple baseline vector (calculated as `activations(word) - mean(activations(baseline_words))`). It helps evaluate how closely your different probes align with a word-based reference across multiple layers.

### Key Features
- **Multi-Probe Support**: Automatically discovers and plots both the aggregated `mass_mass_vector` (Blue Circles) and the individual `PCA_num_*` components (Red Diamonds) from a run folder.
- **Similarity Metrics**: Computes **Cosine Similarity** and **L2 Euclidean Distance** for every probe identified.
- **Smart Labeling**: To reduce clutter, labels are only applied to "points of interest" (e.g., those with `dist > 6` or `abs(cos_sim) > 0.15`).
- **Inverted Visualization**: The X-axis is inverted so that high-quality matches (low distance, high similarity) appear in the top-right.

### Usage Example
```bash
python test_probe_dir_to_simple_baseline.py \
    --probes_folder_path probe_vectors/run_MM_DD_YY_HH_MM/ \
    --concept_word appreciation \
    --min_layer_to_save 10
```

### Command-line Arguments
- `--probes_folder_path` (str, **Required**): Path to the folder containing your `.pt` probe files.
- `--concept_word` (str, **Required**): The specific word to use for the baseline comparison (e.g., `appreciation`).
- `--dataset_name` (str): Dataset for baseline words. Defaults to `simple_data`.
- `--min_layer_to_save` (int): Minimum layer for baseline extraction. Defaults to `16`.
- `--vec_type` (str): Activation type to use (`avg` or `last`). Defaults to `avg`.
- `--save_dir` (str): Directory to save the resulting plot. Defaults to `../../plots/control`.
