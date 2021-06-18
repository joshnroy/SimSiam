#!/bin/bash
#
# This is a half-day long job
#SBATCH -t 10:00:00
#
# Uses 1 GPU
#SBATCH -p gpu --gres=gpu:2
#
# Uses 64 gb ram
#SBATCH --mem=64G
#
# Uses 1 cpu cores
#SBATCH -c 8
#
# Array
#SBATCH --array=1-1

ID=$(($SLURM_ARRAY_TASK_ID))

augs=(RandomResizedCrop RandomHorizontalFlip ColorJitter RandomGrayscale Solarization)
aug=${augs[ID-1]}

exp_type="simsiam_stream51-cifar_time_jittering_deterministic50_staticaugs"
source ~/miniconda3/bin/activate && conda activate simsiam && python3 main.py --config_file="configs/simsiam_stream51.yaml" --data_dir="/users/jroy1/data/jroy1/contrastive/stream_data" --log_dir="../logs/contrastive-logs-${exp_type}-${ID}/" --ckpt_dir=".cache/${exp_type}/" --linear_monitor --temporal_jitter_range=50 --download --wandb --preload_dataset --wandb_group=${exp_type}
