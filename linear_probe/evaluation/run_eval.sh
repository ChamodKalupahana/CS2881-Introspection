#!/bin/bash

# Navigate to the evaluation directory or use relative paths from project root
# We assume the user runs this from the project root.

SAVE_DIR="full_probe_evaluation"
DATASET="dataset/simple_data.json"
N_SAMPLES=50

# 1. Execute Evaluation
# Corrected missing backslash and command structure
python3 linear_probe/evaluation/test_probes.py \
    --dataset "$DATASET" \
    --save_dir "$SAVE_DIR" \
    --n_samples "$N_SAMPLES"

# 2. Identify latest run and generate plots
# This automates the visualization step after the CSV is generated.
LATEST_RUN=$(ls -td "$SAVE_DIR"/run_* 2>/dev/null | head -1)

if [ -n "$LATEST_RUN" ] && [ -f "$LATEST_RUN/final_evaluation_summary.csv" ]; then
    echo "📊 Evaluation summary generated at: $LATEST_RUN"
    echo "📈 Running plot_results.py to generate aggregated visualizations..."
    
    python3 linear_probe/evaluation/plot_results.py \
        --csv_path "$LATEST_RUN/final_evaluation_summary.csv"
        
    echo "✅ Success! Summary and Plot created in $LATEST_RUN"
else
    echo "❌ Error: final_evaluation_summary.csv not found in the latest run directory."
    exit 1
fi