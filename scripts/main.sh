#!/bin/bash
#SBATCH --job-name=main
#SBATCH --account=kempner_sompolinsky_lab
#SBATCH --partition=kempner_h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=0-12:00
#SBATCH --mem=200G
#SBATCH --mail-type=END
#SBATCH --array=0-11
#SBATCH --output=/n/home10/ehahami/work/nov26_experiments/scripts/out/main_%A_%a.out
#SBATCH --error=/n/home10/ehahami/work/nov26_experiments/scripts/out/main_%A_%a.err
#SBATCH --mail-user=elyhahami@college.harvard.edu

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "=========================================="

pip install --upgrade openai

# Define experiment types array
EXPERIMENT_TYPES=(
    "anthropic_reproduce"
    "mcq_knowledge"
    "mcq_distinguish"
    "open_ended_belief"
    "generative_distinguish"
    "injection_strength"
)

# Calculate experiment type and assistant_tokens_only flag
# 6 experiment types × 2 assistant_tokens_only values = 12 total tasks
EXPERIMENT_TYPE_IDX=$((SLURM_ARRAY_TASK_ID / 2))
ASSISTANT_TOKENS_ONLY_IDX=$((SLURM_ARRAY_TASK_ID % 2))

EXPERIMENT_TYPE=${EXPERIMENT_TYPES[$EXPERIMENT_TYPE_IDX]}

if [ $ASSISTANT_TOKENS_ONLY_IDX -eq 0 ]; then
    ASSISTANT_TOKENS_ONLY="--assistant_tokens_only"
    ASSISTANT_TOKENS_ONLY_LABEL="true"
else
    ASSISTANT_TOKENS_ONLY="--no_assistant_tokens_only"
    ASSISTANT_TOKENS_ONLY_LABEL="false"
fi

echo "Running experiment type: $EXPERIMENT_TYPE"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Experiment type index: $EXPERIMENT_TYPE_IDX"
echo "Assistant tokens only: $ASSISTANT_TOKENS_ONLY_LABEL"
echo ""

module purge
module load python
mamba deactivate
mamba activate /n/holylabs/LABS/sompolinsky_lab/Everyone/xupan/envs/gpt

# Source .bashrc to load environment variables (HF_HOME, HUGGING_FACE_HUB_TOKEN)
source ~/.bashrc

# Create cache directory
mkdir -p /tmp/ehahami/cache

which python

cd /n/home10/ehahami/work/nov26_experiments

# Default parameters (can be overridden by command line)
LAYERS="9 12 15 18"
COEFFS="4 9 16"

# Use full path to Python in mamba environment
# Run with the selected experiment type
/n/holylabs/LABS/sompolinsky_lab/Everyone/xupan/envs/gpt/bin/python main.py \
    --type "$EXPERIMENT_TYPE" \
    --layers $LAYERS \
    --coeffs $COEFFS \
    $ASSISTANT_TOKENS_ONLY

echo ""
echo "=========================================="
echo "Completed experiment type: $EXPERIMENT_TYPE"
echo "Finished at: $(date)"
echo "=========================================="