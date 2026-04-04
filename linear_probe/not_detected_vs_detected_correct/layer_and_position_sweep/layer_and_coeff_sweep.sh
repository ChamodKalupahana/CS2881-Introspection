#!/bin/bash

# Configuration
DATASET="brysbaert_abstract_nouns.json"
N_SAMPLES=10
VECTOR_TYPE="average"
MAX_TOKENS=50

# Default Sweep Grids (Broad)
COEFFS=(2 4 6 8)
LAYERS=(10 14 18 22 24 26 28)
MODE="zoomed"

# Check for flags
TEST_FLAG=""
for arg in "$@"; do
    if [ "$arg" == "--test" ]; then
        TEST_FLAG="--test"
        echo "🛠️  Running in TEST MODE (Fake Data)"
    elif [ "$arg" == "--zoomed" ]; then
        MODE="zoomed"
        COEFFS=(3 4 5 6)
        LAYERS=(10 11 12 13 14 15 16 17 18)
    fi
done

echo "🔍 Sweep Mode: $MODE"

# Create a unique directory for this entire sweep session
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SWEEP_DIR="saved_activations/grid_sweep_${TIMESTAMP}"
[[ -n "$TEST_FLAG" ]] && SWEEP_DIR="${SWEEP_DIR}_TEST"
mkdir -p "$SWEEP_DIR"

echo "🚀 Starting Grid Sweep Session: $SWEEP_DIR"
echo "📊 Layers [${LAYERS[*]}] x Coeffs [${COEFFS[*]}]"

for coeff in "${COEFFS[@]}"; do
    for layer in "${LAYERS[@]}"; do
        RUN_NAME="L${layer}_C${coeff}"
        
        echo -e "\n------------------------------------------------------------"
        echo "🏃 Run: $RUN_NAME | Layer: $layer | Coeff: $coeff"
        echo "------------------------------------------------------------"
        
        python save_activations_by_layer_and_position.py \
            --dataset "$DATASET" \
            --layer "$layer" \
            --coeff "$coeff" \
            --n_samples "$N_SAMPLES" \
            --vector_type "$VECTOR_TYPE" \
            --max_new_tokens "$MAX_TOKENS" \
            --run_name "$RUN_NAME" \
            --save_dir "$SWEEP_DIR" \
            --no_save \
            $TEST_FLAG
            
        if [ $? -ne 0 ]; then
            echo "⚠️  Run failed for Layer $layer Coeff $coeff. Continuing..."
        fi
    done
done

echo -e "\n📈 Generating Final Plots for session: $SWEEP_DIR"
python plot_layer_and_coeff_sweep_results.py --base_dir "$SWEEP_DIR"

echo -e "\n✅ Grid Sweep Complete! View results in: /workspace/CS2881-Introspection/plots/linear_probe/not_detected_vs_detected_correct/layer_and_coeff_sweep/sweep_results.png"

# python plot_layer_and_coeff_sweep_results.py --xlim 9 29 --base_dir saved_activations/grid_sweep_20260404_202711