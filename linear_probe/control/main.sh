#!/bin/bash

# Configuration
CONCEPT="appreciation"
MIN_LAYER=10

# 1. Save activations and concept vectors (generates a timestamped run directory)
python save_activations_and_concept.py \
    --dataset_name "complex_data" \
    --concept_name "$CONCEPT" \
    --min_layer_to_save "$MIN_LAYER"

# 2. Identify the latest run directory in probe_vectors
# ls -td sorts by modification time (descending), head -n 1 picks the first (latest)
LATEST_RUN_DIR=$(ls -td probe_vectors/run_*/ | head -n 1)

if [ -z "$LATEST_RUN_DIR" ]; then
    echo "❌ Error: No run directory found in probe_vectors/"
    exit 1
fi

PROBE_FILE="${LATEST_RUN_DIR}mass_mass_vector_${CONCEPT}.pt"

echo "🎯 Using latest probe path: $PROBE_FILE"

# 3. Run the baseline comparison script
python test_probe_dir_to_simple_baseline.py \
    --probe_path "$PROBE_FILE" \
    --concept_word "$CONCEPT" \
    --min_layer_to_save "$MIN_LAYER"