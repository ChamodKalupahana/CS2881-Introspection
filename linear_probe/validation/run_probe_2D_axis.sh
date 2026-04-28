#!/bin/bash

# 1. Define Run Configuration
# Generating a timestamped RUN_NAME so all scripts use the same directory
RUN_NAME="run_$(date +%m_%d_%y_%H_%M)"
SAVE_ROOT="saved_activations"
RUN_DIR="${SAVE_ROOT}/${RUN_NAME}"

echo "🚀 Starting Introspection Activation Sweep: ${RUN_NAME}"

# 2. Capture Layer and Position Activations
# We provide the explicit --run_name to avoid date skews between shell and python
python3 save_activations_by_layer_and_position.py \
    --dataset brysbaert_abstract_nouns.json \
    --layer 14 \
    --coeff 5.0 \
    --vector_type average \
    --max_new_tokens 50 \
    --run_name "${RUN_NAME}" \

python3 probe_2D_axis.py \
    --run_dir "${RUN_NAME}" \
    --probe1 calibation_correct_vs_detected_correct/probe_vectors/MM_L18_P0_dPrim22.88_dVal1.37.pt \
    --probe2 not_detected_vs_detected_correct/layer_and_position_sweep/probe_vectors/MM_L26_P0_dPrim1.69_dVal1.36.pt
