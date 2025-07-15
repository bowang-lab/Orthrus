#!/bin/bash

#SBATCH --job-name=orthrus_train
#SBATCH --partition=gpu,morrisq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:4
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=24:00:00
#SBATCH --output=/home/fradkinp/Documents/01_projects/Orthrus_private/slurm/slurm_out/orthrus_train.%A_%a.out
#SBATCH --error=/home/fradkinp/Documents/01_projects/Orthrus_private/slurm/slurm_out/orthrus_train.%A_%a.err

# prepare your environment here
echo `date`: Job $SLURM_JOB_ID is allocated resource
echo "Starting task $SLURM_ARRAY_TASK_ID"

# virtual env
eval "$(conda shell.bash hook)"
conda activate orthrus_private

cd /home/fradkinp/Documents/01_projects/Orthrus_private/orthrus


# Read command-line arguments for each flag
NOTE=${1:-"1_ablation_u08_mlm_cl"}
DATA_CONFIG=${2:-"new_splice_to_toga_combined_15_mask"}
MODEL_CONFIG=${3:-"mamba_deep_mlm"}
PROJECTOR_CONFIG=${4:-"default_512"}
OPTIMIZER_CONFIG=${5:-"underweight_ortho_0.8"}
TRAIN_CONFIG=${6:-"a100_ssm_512_6_deep_mlm_05_95_ablation"}

# Run the Python script with modified flags
srun python train.py \
  --note="$NOTE" \
  --data_config="$DATA_CONFIG" \
  --model_config="$MODEL_CONFIG" \
  --projector_config="$PROJECTOR_CONFIG" \
  --optimizer_config="$OPTIMIZER_CONFIG" \
  --train_config="$TRAIN_CONFIG"

