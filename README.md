# Emergent Introspection in LLMs

This repository implements experiments on emergent introspection in large language models, following the methodology from Anthropic's [Emergent Introspection paper](https://transformer-circuits.pub/2025/introspection/index.html). The core idea is to test whether models can detect and report on "injected thoughts" - concept vectors that we add to their hidden states during inference. Read the full paper here: https://arxiv.org/abs/2512.12411.

## Overview

We inject concept vectors (representations of specific concepts like "Dust", "betrayal", "fibonacci_numbers") into a model's hidden states at various layers and with different strengths, then ask the model to introspect about what it's "thinking" about. The model's responses are evaluated using an LLM-as-a-judge approach to determine if it successfully detected and identified the injected concept.

## Project Structure

```
nov26_experiments/
├── dataset/                    # Raw data for computing concept vectors
│   ├── simple_data.json       # Concrete nouns (from Anthropic paper appendix)
│   └── complex_data.json      # Abstract concepts with positive/negative examples
├── saved_vectors/              # Pre-computed concept vectors
│   ├── llama/                 # Vectors for Llama models
│   └── qwen/                  # Vectors for Qwen models
├── new_results/                # Experiment results (CSV and Parquet)
├── plots/                      # Plotting scripts and generated figures
│   ├── plots.py               # Script to generate success rate plots
│   └── *.png                   # Generated plot images
├── abilityToRevert/            # Experiments testing if models can revert/undo injected thoughts
├── multipleInjections/         # Experiments testing multiple simultaneous concept injections
├── scripts/                     # Slurm batch job scripts
│   ├── main.sh                # Main experiment runner (array job)
│   └── out/                    # Job output and error logs
├── compute_concept_vector_utils.py  # Utilities for computing concept vectors
├── save_vectors.py             # Script to compute and save vectors for all layers
├── inject_concept_vector.py    # Core injection mechanism
├── main.py                     # Main experiment script
├── all_prompts.py              # All prompts and grader templates
└── api_utils.py                # LLM judge interface
```

## Quick Start

### 1. Compute Concept Vectors

First, compute concept vectors for all layers and concepts. You can run this directly or submit as a batch job:

```bash
python save_vectors.py
```

Or submit as a batch job (you'll need to create a `save_vectors.sh` script or run it interactively):

```bash
sbatch scripts/save_vectors.sh  # If you create this script
```

This will:
- Load the model (default: `meta-llama/Meta-Llama-3.1-8B-Instruct`)
- Process both `simple_data` and `complex_data` datasets
- Extract activations at each layer for each concept
- Save vectors to `saved_vectors/llama/` as `.pt` files

Vectors are saved with filenames like `{concept}_{layer}_{vec_type}.pt` where `vec_type` is either `avg` (average across prompt tokens) or `last` (final token activation).

### 2. Run Experiments

Run the full experiment suite:

```bash
sbatch scripts/main.sh
```

This submits an array job that runs all 6 experiment types × 2 `assistant_tokens_only` settings = 12 parallel tasks. Each task:
- Loads concept vectors from `saved_vectors/llama/`
- Sweeps over layers (default: 9, 12, 15, 18) and coefficients (default: 4, 9, 16)
- Injects vectors and collects model responses
- Evaluates responses using LLM judges
- Saves results incrementally to `new_results/output_{experiment_type}.csv`

### 3. Generate Plots

After experiments complete, generate success rate plots:

```bash
python plots/plots.py
```

This reads from `new_results/` and generates plots in `plots/` showing success rates as a function of layer, with separate lines for each (coefficient, vec_type) combination.

## Experiment Types

We test six different prompt types to probe introspection capabilities:

1. **`anthropic_reproduce`**: Direct reproduction of Anthropic's experiment - asks "Do you detect an injected thought? If so, what is it about?"
2. **`mcq_knowledge`**: Multiple choice with 10 options (the injected concept + 9 distractors)
3. **`mcq_distinguish`**: Multiple choice with 2 options (the injected concept + 1 distractor)
4. **`open_ended_belief`**: Open-ended question "What concept are you thinking about right now?"
5. **`generative_distinguish`**: Asks model to distinguish between two phenomena (one true, one false)
6. **`injection_strength`**: Asks model to categorize injection strength (Weak/Moderate/Strong/Very strong)

## Evaluation

Responses are evaluated using an LLM-as-a-judge approach (GPT-5-nano-2025-08-07) with structured grading prompts from Anthropic's paper. We evaluate multiple criteria:

- **Coherence**: Does the response coherently describe mental states?
- **Thinking about word**: Does the model report thinking about the injected concept?
- **Affirmative response**: Does the model claim to detect an injected thought?
- **Affirmative response + correct identification**: Does the model both detect and correctly identify the concept?
- **MCQ correct**: Did the model select the correct multiple-choice answer?
- **Injection strength correct**: Did the model correctly categorize the injection strength?

Success rates are computed as conjunctions of relevant criteria (e.g., for MCQ: `coherence AND mcq_correct`).

## Key Files

### `compute_concept_vector_utils.py`
Computes concept vectors from datasets. For `simple_data`, it computes `vector(concept) - mean(vector(baseline_words))`. For `complex_data`, it computes `mean(positive_examples) - mean(negative_examples)`. Returns both "last token" and "average" variants.

### `inject_concept_vector.py`
Core injection mechanism. Registers a forward hook at the target layer that adds `coeff * normalized_vector` to hidden states. Supports injecting at all tokens or only during assistant token generation.

### `main.py`
Main experiment script. Loads vectors, runs injections across all combinations of (concept, layer, vec_type, coefficient), collects responses, evaluates with judges, and saves results. Also generates basic plots.

### `all_prompts.py`
Centralizes all prompts and grader templates. Makes it easy to modify prompts or add new experiment types.

### `api_utils.py`
Interface to OpenAI API for LLM-as-a-judge evaluation. Handles different grading types and parses YES/NO responses.

### `plots/plots.py`
Plotting script that generates success rate plots for all experiment types. Reads from `new_results/` and saves plots to `plots/`.

## Results Format

Results are saved as CSV files in `new_results/` with columns:
- `concept`, `vec_type`, `layer`, `coeff`, `type`, `assistant_tokens_only`
- Judge results: `coherence_judge`, `thinking_about_word_judge`, `affirmative_response_judge`, etc.
- `response`: The raw model response
- `expected_strength_category`: For injection_strength experiments

Results are saved incrementally during experiments, so you can check progress by reading the CSV files.

## Customization

### Changing Layers/Coefficients

Edit `scripts/main.sh` to modify the default layers and coefficients:

```bash
LAYERS="9 12 15 18"
COEFFS="4 9 16"
```

### Running a Single Experiment Type

You can run `main.py` directly:

```bash
python main.py --type anthropic_reproduce --layers 12 15 18 --coeffs 5 10 15 --assistant_tokens_only
```

### Adding New Experiment Types

1. Add a message template function in `all_prompts.py` (e.g., `get_my_new_type_messages()`)
2. Add the type to the choices in `main.py`'s argument parser
3. Add a case in `test_vector_multiple_choice()` to use your new messages
4. Define success criteria in `plots/plots.py` if you want automatic plotting

## Data Sources

- **`simple_data.json`**: Taken directly from Anthropic's paper appendix. Contains concrete nouns like "Dust", "Satellites" paired with baseline words.
- **`complex_data.json`**: Synthetically generated by prompting an LLM to create more examples. Contains abstract concepts like "fibonacci_numbers", "betrayal", "appreciation" with positive/negative sentence pairs.

## Notes

- The code uses `torch.manual_seed(2881)` for reproducibility in MCQ option shuffling.
- Vector injection normalizes vectors to unit length before scaling by the coefficient.
- The `injection_start_token` parameter allows fine-grained control over when injection begins (e.g., only after "Trial 1" in the prompt).
- Results are saved both as CSV (human-readable) and Parquet (efficient, preserves types).
- Job output and error logs are saved in `scripts/out/` with filenames like `main_{job_id}_{array_id}.out` and `main_{job_id}_{array_id}.err`.

## Dependencies

- PyTorch
- Transformers (HuggingFace)
- OpenAI API (for LLM judge)
- pandas, matplotlib (for analysis/plotting)
- pyarrow (optional, for Parquet export)

## References

- [Anthropic's Emergent Introspection Paper](https://transformer-circuits.pub/2025/introspection/index.html)
- [Anthropic's SDF Paper](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/) (for experiment type inspiration)
