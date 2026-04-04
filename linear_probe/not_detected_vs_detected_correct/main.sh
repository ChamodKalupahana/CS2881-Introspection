#!/bin/bash

# List of models to validate
MODELS=("gpt-4.1-nano" "gpt-5-nano" "gpt-4.1-mini" "gpt-4o-mini" "gpt-5.4-nano")

# Check for test mode
TEST_FLAG=""
if [[ "$1" == "--test" ]]; then
    TEST_FLAG="--test"
    echo "🧪 RUNNING IN TEST MODE (Mock Results)"
fi

# Generate a common timestamp for this batch run
TIMESTAMP=$(date +%m_%d_%y_%H_%M)
RESULTS_PATHS=()

# Run validation for each model
for model in "${MODELS[@]}"; do
    RUN_NAME="${model}_${TIMESTAMP}"
    echo ""
    echo "========================================================================="
    echo "VALIDATING JUDGE MODEL: $model (Run: $RUN_NAME)"
    echo "========================================================================="
    
    python3 validate_llm_judges.py --mode separate --model "$model" --run_name "$RUN_NAME" $TEST_FLAG
    
    # Track the results.json path
    RESULTS_PATHS+=("validation_llm_judges_runs/${RUN_NAME}/results.json")
done

# Run the plotting script
echo ""
echo "========================================================================="
echo "GENERATING ACCURACY COMPARISON PLOT"
echo "========================================================================="
python3 plot_llm_judge_accuracy.py "${RESULTS_PATHS[@]}"

echo ""
echo "✅ All runs complete. Comparison plot saved."