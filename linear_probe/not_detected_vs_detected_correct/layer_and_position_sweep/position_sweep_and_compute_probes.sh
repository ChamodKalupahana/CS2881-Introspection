#!/bin/bash

# 1. Define Run Configuration
# Generating a timestamped RUN_NAME so all scripts use the same directory
RUN_NAME="run_$(date +%m_%d_%y_%H_%M)"
SAVE_ROOT="saved_activations"
RUN_DIR="${SAVE_ROOT}/${RUN_NAME}"

echo "🚀 Starting Introspection Activation Sweep: ${RUN_NAME}"

# 2. Capture Layer and Position Activations
# We provide the explicit --run_name to avoid date skews between shell and python
python save_activations_by_layer_and_position.py \
    --dataset brysbaert_abstract_nouns.json \
    --layer 14 \
    --coeff 5.0 \
    --vector_type average \
    --max_new_tokens 50 \
    --run_name "${RUN_NAME}" \
    --n_samples 5

# 3. Analyze Mass-Mean discriminability across all layers/positions
# This helps identify the 'best position' for steering (e.g., position 0)
echo "📊 Running Mass-Mean Sweep..."
python layer_and_position_sweep_mass_mean_vectors.py --run_dir "${RUN_DIR}"

# 4. Perform Detailed PCA Analysis & Vector Export for the best position
# We focus on --position 0 as it's typically the most discriminative
echo "🔬 Running Detailed PCA & Probe Export for Position 0..."
python compute_mass_mean_vectors_and_PCA.py \
    --run_dir "${RUN_DIR}" \
    --position 0 \
    --top_right_only

python compute_mass_mean_vectors_and_PCA.py \
    --run_dir "saved_activations/run_04_05_26_13_31" \
    --position 0 \
    --top_right_only