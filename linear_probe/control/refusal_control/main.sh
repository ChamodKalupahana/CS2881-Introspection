#!/bin/bash

# 1. Extract and Save Refusal Activations and Probes
# This will save files to PROJECT_ROOT/probe_vectors/refusal/run_MM_DD_HH_MM
python save_refusal_activations.py --num_prompts 100

# 2. Get the latest run folder
# We check the central probe_vectors/refusal directory (relative to project root)
# From this folder, projector root is ../../../
LATEST_RUN=$(ls -td ../../../probe_vectors/refusal/run_*/ | head -n 1)

echo "🚀 Comparing latest run: $LATEST_RUN"

# 3. Compare with Ground Truth
# We pass the absolute or relative path to the latest discovered run
python test_probe_dir_to_ground_truth.py --probes_folder_path "$LATEST_RUN"
linear_probe/control/refusal_control/probe_vectors/refusal/run_04_03_19_46