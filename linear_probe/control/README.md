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
