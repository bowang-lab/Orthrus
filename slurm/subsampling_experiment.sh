#!/bin/bash

# =========================================================================================
# SLURM DIRECTIVES
# These lines are instructions for the SLURM scheduler.
# =========================================================================================
#
# --job-name: The name of your job, which appears in queue listings.
#SBATCH --job-name=finetuning
#
# --output & --error: The files where standard output and errors are written.
# %A is the job ID, %a is the array task ID.
#SBATCH --output=./slurm_out/orthrus_finetune.%A_%a.out
#SBATCH --error=./slurm_out/orthrus_finetune.%A_%a.err
#
# --partition: The queue or partition to submit the job to.
#SBATCH --partition=morrisq,gpu
#
# --gres: Generic RESource request. Here we ask for 1 GPUs.
#SBATCH --gres=gpu:h100:1
#
# --time: The maximum time the job will run for. Format is HH:MM:SS.
#SBATCH --time 16:00:00
#
# --cpus-per-task: The number of CPU cores to allocate for this job.
#SBATCH --cpus-per-task=8
#
# --mem-per-cpu: The amount of RAM to allocate per CPU core.
#SBATCH --mem-per-cpu=16GB
#
# --nodes & --ntasks-per-node: We are requesting 1 node and 1 task per node.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#
# --exclude: Do not run on this specific node.
#SBATCH --exclude=isch009
#
# --export=ALL: This makes sure the environment variables from your submission shell
# are available in the job's shell.
#SBATCH --export=ALL
#
# NOTE: The --array directive is NOT set here. You must set it when you submit the job,
# based on the model class you choose. The script will tell you the correct range.
# Example: sbatch --array=0-26 slurm/subsampling_experiment.sh


# =========================================================================================
# SCRIPT CONFIGURATION
# =========================================================================================

# --- Step 1: Testing Configuration ---
# To test this script before submitting to SLURM, use the settings below.
#
# DRY_RUN:
#   - "true":  Print the python commands that would be run, without executing them.
#   - "false": Actually execute the python commands.
#
# TEST_TASK_ID:
#   - The array task ID you want to simulate (e.g., 0 for the first job, 7 for the eighth).
#
DRY_RUN="true"
TEST_TASK_ID="0"

# --- Step 2: Choose which model class to run ---
# TODO: Change this value to one of: "resnet", "saluki", "mamba", or "contrastive"
MODEL_CLASS_TO_RUN="contrastive"


# --- Step 3: Define Experiment Parameters ---

# An array of the random seeds to use for training.
SEEDS=(0)

# An array of the different training sample sizes to test.
SAMPLE_SIZES=(
    3000
    1000
    300
    100
    30
)
# SAMPLE_SIZES=(3000 1000 300 100 30)

# A parallel array that defines the training configuration for each sample size.
# The order MUST match the order in SAMPLE_SIZES.
# For example, SAMPLE_SIZES[0] (3000) will use TRAIN_CONFIGS[0] ("bs_64_3000_steps").
TRAIN_CONFIGS=(
    "bs_64_3000_steps"
    "bs_64_1000_steps"
    "bs_64_1000_steps"
    "bs_32_2000_steps"
    "bs_16_2000_steps"
)

# Special training configurations for ECLIP datasets.
ECLIP_TRAIN_CONFIGS=(
    "bs_64_500_steps"
    "bs_64_500_steps"
    "bs_64_500_steps"
    "bs_32_500_steps"
    "bs_16_200_steps"
)


# =========================================================================================
# MODEL CONFIGURATIONS
# Each line is a string with 5 parts:
# 1. model_config  2. data_config  3. optimizer_config  4. (ignored)  5. projector_config
# =========================================================================================

DILATED_RESNET_CONFIGS=(
    "dilated_resnet_small go_mf high_reg_1e-3 bs_64_10k dilated_resnet_small_projector"
    "dilated_resnet_small rnahl_human high_reg_1e-3 bs_64_10k dilated_resnet_small_projector"
    "dilated_resnet_small rnahl_mouse high_reg_1e-3 bs_64_10k dilated_resnet_small_projector"
    "dilated_resnet_small rna_loc_fazal high_reg_1e-3 bs_64_10k dilated_resnet_small_projector"
    "dilated_resnet_small rna_loc_ietswaart high_reg_1e-3 bs_64_10k dilated_resnet_small_projector"
    "dilated_resnet_small prot_loc high_reg_1e-4 bs_64_10k dilated_resnet_small_projector"
    "dilated_resnet_small mrl_sugimoto high_reg_1e-4 bs_64_10k dilated_resnet_small_projector"
    "dilated_resnet_small eclip_binding_k562 high_reg_1e-3 bs_64_3k dilated_resnet_small_projector"
    "dilated_resnet_small eclip_binding_hepg2 high_reg_1e-3 bs_64_3k dilated_resnet_small_projector"
)

SALUKI_CONFIGS=(
    "saluki go_mf saluki bs_64_10k saluki_projector"
    "saluki rnahl_human saluki bs_64_10k saluki_projector"
    "saluki rnahl_mouse saluki bs_64_10k saluki_projector"
    "saluki rna_loc_fazal saluki bs_64_10k saluki_projector"
    "saluki rna_loc_ietswaart saluki bs_64_10k saluki_projector"
    "saluki prot_loc saluki bs_64_10k saluki_projector"
    "saluki mrl_sugimoto saluki bs_64_10k saluki_projector"
    "saluki eclip_binding_k562 saluki bs_64_3k saluki_projector"
    "saluki eclip_binding_hepg2 saluki bs_64_3k saluki_projector"
)

MAMBA_CONFIGS=(
    "mamba_base go_mf high_reg_1e-3 bs_64_10k mamba_base_projector"
    "mamba_base rnahl_human high_reg_1e-3 bs_64_10k mamba_base_projector"
    "mamba_base rnahl_mouse high_reg_1e-3 bs_64_10k mamba_base_projector"
    "mamba_base rna_loc_fazal high_reg_1e-3 bs_64_10k mamba_base_projector"
    "mamba_base rna_loc_ietswaart high_reg_1e-3 bs_64_10k mamba_base_projector"
    "mamba_base prot_loc high_reg_1e-4 bs_64_10k mamba_base_projector"
    "mamba_base mrl_sugimoto high_reg_1e-4 bs_64_10k mamba_base_projector"
    "mamba_base eclip_binding_k562 high_reg_1e-4 bs_64_3k mamba_base_projector"
    "mamba_base eclip_binding_hepg2 high_reg_1e-4 bs_64_3k mamba_base_projector"
)

CONTRASTIVE_CONFIGS=(
    "contrastive_only_small go_mf no_wd_1e-3 bs_64_10k mamba_base_projector"
    "contrastive_only_small rnahl_human no_wd_1e-3 bs_64_10k mamba_base_projector"
    "contrastive_only_small rnahl_mouse no_wd_1e-3 bs_64_10k mamba_base_projector"
    "contrastive_only_small rna_loc_fazal no_wd_1e-4 bs_64_10k mamba_base_projector"
    "contrastive_only_small rna_loc_ietswaart no_wd_1e-4 bs_64_10k mamba_base_projector"
    "contrastive_only_small prot_loc no_wd_1e-4 bs_64_10k mamba_base_projector"
    "contrastive_only_small mrl_sugimoto no_wd_3e-4_clip_0.5 bs_64_10k mamba_base_projector"
    "contrastive_only_small eclip_binding_k562 no_wd_1e-4 bs_64_3k mamba_base_projector"
    "contrastive_only_small eclip_binding_hepg2 no_wd_1e-4 bs_64_3k mamba_base_projector"
)


# =========================================================================================
# JOB SETUP
# This section selects the correct configurations and calculates the SLURM array size.
# =========================================================================================

# The 'case' statement is like a switch statement in other languages.
# It selects the array of configurations based on the MODEL_CLASS_TO_RUN variable.
case "$MODEL_CLASS_TO_RUN" in
    "resnet")
        CONFIGS=("${DILATED_RESNET_CONFIGS[@]}")
        ;;
    "saluki")
        CONFIGS=("${SALUKI_CONFIGS[@]}")
        ;;
    "mamba")
        CONFIGS=("${MAMBA_CONFIGS[@]}")
        ;;
    "contrastive")
        CONFIGS=("${CONTRASTIVE_CONFIGS[@]}")
        ;;
    *)
        echo "Error: MODEL_CLASS_TO_RUN must be one of 'resnet', 'saluki', 'mamba', or 'contrastive'."
        exit 1
        ;;
esac

# Calculate the total number of jobs needed for the SLURM array.
# It's the number of model configurations multiplied by the number of seeds.
NUM_CONFIGS=${#CONFIGS[@]}
NUM_SEEDS=${#SEEDS[@]}
MAX_ARRAY_ID=$((NUM_CONFIGS * NUM_SEEDS - 1))

# This message prints when you run the script, telling you how to submit it.
echo "Job configuration:"
echo "  Model class: $MODEL_CLASS_TO_RUN"
echo "  Number of model configs: $NUM_CONFIGS"
echo "  Number of seeds: $NUM_SEEDS"
echo "  Total array jobs: $((MAX_ARRAY_ID + 1))"
echo "  Submit this script with: sbatch --array=0-$MAX_ARRAY_ID $0"


# =========================================================================================
# JOB EXECUTION
# This section runs for each task in the SLURM job array.
# =========================================================================================

# --- Determine the configuration for THIS specific job task ---

# Check if we are running inside a SLURM job or locally for a test.
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    # The variable is not set, so we are running locally.
    echo "--- RUNNING IN LOCAL TEST MODE ---"
    CURRENT_TASK_ID=$TEST_TASK_ID
else
    # The variable is set, so we are in a real SLURM job.
    echo "--- RUNNING IN SLURM JOB MODE (Task: $SLURM_ARRAY_TASK_ID) ---"
    CURRENT_TASK_ID=$SLURM_ARRAY_TASK_ID
fi

echo "Processing for Task ID: $CURRENT_TASK_ID"

# Use the determined task ID for all calculations.
CONFIG_INDEX=$((CURRENT_TASK_ID / NUM_SEEDS))
SEED_INDEX=$((CURRENT_TASK_ID % NUM_SEEDS))
SEED=${SEEDS[$SEED_INDEX]}

# Get the specific configuration line from the CONFIGS array.
CONFIG_LINE=${CONFIGS[$CONFIG_INDEX]}

# The `read` command splits the CONFIG_LINE string into separate variables.
# -r: prevents backslash interpretation
# -a: reads the words into an array named 'config_parts'
read -r -a config_parts <<< "$CONFIG_LINE"

MODEL_CONFIG=${config_parts[0]}
DATA_CONFIG=${config_parts[1]}
OPTIMIZER_CONFIG=${config_parts[2]}
# The original train_config (config_parts[3]) is ignored because we will
# set it inside the loop based on the sample size.
PROJECTOR_CONFIG=${config_parts[4]}


# --- Setup Environment ---
echo "Loading conda environment and changing directory..."
eval "$(conda shell.bash hook)"
conda activate mrna_bench
cd /home/fradkinp/Documents/01_projects/orthrus_private/orthrus
echo "Conda environment activated. Current directory: $(pwd)"


# --- Run the experiments for different data subsets ---
# Decide which training configs to use based on the dataset
if [[ $DATA_CONFIG == eclip* ]]; then
    echo "Using ECLIP-specific training configurations."
    CURRENT_TRAIN_CONFIGS=("${ECLIP_TRAIN_CONFIGS[@]}")
else
    CURRENT_TRAIN_CONFIGS=("${TRAIN_CONFIGS[@]}")
fi

# This loop iterates from 0 to 4 (for the 5 items in SAMPLE_SIZES).
for i in "${!SAMPLE_SIZES[@]}"; do
    N_SAMPLES=${SAMPLE_SIZES[$i]}
    TRAIN_CONFIG=${CURRENT_TRAIN_CONFIGS[$i]}

    echo "----------------------------------------------------"
    echo "Running with:"
    echo "  Model Config: $MODEL_CONFIG"
    echo "  Data Config: $DATA_CONFIG"
    echo "  Optimizer Config: $OPTIMIZER_CONFIG"
    echo "  Projector Config: $PROJECTOR_CONFIG"
    echo "  Train Config: $TRAIN_CONFIG"
    echo "  Sample Size: $N_SAMPLES"
    echo "  Seed: $SEED"
    echo "----------------------------------------------------"

    # Finally, run the Python script with all the determined parameters.
    if [ "$DRY_RUN" = "true" ]; then
        echo "--- DRY RUN (Task ID: $CURRENT_TASK_ID) ---"
        echo "Would execute:"
        echo "python finetune.py \\"
        echo "    --data_config \"${DATA_CONFIG}\" \\"
        echo "    --model_config \"${MODEL_CONFIG}\" \\"
        echo "    --projector_config \"${PROJECTOR_CONFIG}\" \\"
        echo "    --optimizer_config \"${OPTIMIZER_CONFIG}\" \\"
        echo "    --train_config \"${TRAIN_CONFIG}\" \\"
        echo "    --subset_n_samples \"${N_SAMPLES}\" \\"
        echo "    --seed_override \"${SEED}\" \\"
        echo "    --note \"${MODEL_CLASS_TO_RUN}_${DATA_CONFIG}_${N_SAMPLES}_samples_seed_${SEED}\""
    else
        python finetune.py \
            --data_config "${DATA_CONFIG}" \
            --model_config "${MODEL_CONFIG}" \
            --projector_config "${PROJECTOR_CONFIG}" \
            --optimizer_config "${OPTIMIZER_CONFIG}" \
            --train_config "${TRAIN_CONFIG}" \
            --subset_n_samples "${N_SAMPLES}" \
            --seed_override "${SEED}" \
            --note "${MODEL_CLASS_TO_RUN}_${DATA_CONFIG}_${N_SAMPLES}_samples_seed_${SEED}"
    fi

    # Check if the python command failed.
    if [ $? -ne 0 ] && [ "$DRY_RUN" = "false" ]; then
        echo "Error: The command for ${N_SAMPLES} samples failed."
    fi
done

echo "Task $CURRENT_TASK_ID finished."
wait 