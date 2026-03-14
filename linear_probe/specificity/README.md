# Concept Specificity Experiments

This directory contains scripts for assessing the **specificity** of concept vectors. The goal is to determine whether a concept vector injected into a model's activations is detected correctly by the model, or if it's confused with other related, opposite, or unrelated concepts.

## Core Components

### [save_vectors_orthogonal.py](file:///workspace/CS2881-Introspection/linear_probe/specificity/save_vectors_orthogonal.py)

This is the primary script for running concept injection experiments and capturing activations categorized by the model's detection outcome.

#### Key Features:
- **Concept Injection**: Injects computed concept vectors (avg or last token) at specified layers with adjustable coefficients.
- **Activation Capture**: Records hidden states from the injection layer to the final layer.
- **3-Stage LLM Judge Pipeline**: Uses an LLM (GPT-4-nano) to classify the model's response through three distinct stages:
    1.  **Coherence Check**: Filters out gibberish or non-intelligible responses.
    2.  **Affirmative Detection Check**: Determines if the model claims to have noticed an anomaly or injected thought.
    3.  **Category Classification**: Classifies the detected concept relative to the injected one into categories:
        - `detected_correct`: Accurate identification of the concept.
        - `detected_parallel`: Identification of a related concept in the same domain.
        - `detected_opposite`: Identification of the antonym/opposite concept.
        - `detected_orthogonal`: Detection of a thought, but identifying it as something completely unrelated.
        - `detected_unknown`: Claiming detection but failing to name a specific concept.
        - `not_detected`: No claim of detection.
        - `incoherent`: Response was nonsensical.

## Usage

```bash
# Run specificity test for a specific layer and coefficient
python3 linear_probe/specificity/save_vectors_orthogonal.py --layers 18 --coeffs 10 --datasets simple_data
```

### Arguments:
- `--model`: The LLM to test (default: `Llama-3.1-8B-Instruct`).
- `--layers`: Layer(s) at which to inject the concept vector.
- `--coeffs`: Injection strength coefficients to sweep.
- `--vec_type`: Use `avg` or `last` token vector.
- `--datasets`: Concept datasets to use (e.g., `simple_data`).
- `--capture_all_layers`: Capture activations from the injection layer through to the final layer.

## Output Structure

Results are saved in the configured `save_dir`:
```text
run_MM_DD_YY_HH_MM/
  run.log                 # Detailed execution and judge logs
  detected_correct/       # .pt files for correctly detected injections
  detected_parallel/      # .pt files for related detections
  ...
```
Each `.pt` file contains the concept name, dataset, classification category, injection metadata, the model response, and the captured activations.
