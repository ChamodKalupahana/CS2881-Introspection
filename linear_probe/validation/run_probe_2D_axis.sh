#!/bin/bash

# 1. Define Run Configuration
# Generating a timestamped RUN_NAME so all scripts use the same directory
RUN_NAME="run_$(date +%m_%d_%y_%H_%M)"
SAVE_ROOT="saved_activations"
RUN_DIR="${SAVE_ROOT}/${RUN_NAME}"

echo "🚀 Starting Introspection Activation Sweep: ${RUN_NAME}"

# 2. Capture Layer and Position Activations
# We provide the explicit --run_name to avoid date skews between shell and python
python save_activations_by_layer_and_position_with_calibation.py \
    --dataset brysbaert_abstract_nouns.json \
    --layer 14 \
    --coeff 5.0 \
    --vector_type average \
    --max_new_tokens 50 \
    --run_name "${RUN_NAME}" \
    --n_samples 5

python probe_2D_axis.py \
    --run_dir "${RUN_NAME}" \
    --probe1  \
    --probe2  \
