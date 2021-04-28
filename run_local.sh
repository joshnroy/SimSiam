#!/bin/bash

for i in 50 0;
	do for j in 64;
	    do export CUDA_VISIBLE_DEVICES=0,1 && export WANDB_RUN_GROUP=jittering_deterministic${i}_resolution${j} && python main.py --config_file="configs/simsiam_stream51.yaml" --data_dir="../stream_data/" --log_dir="../logs/contrastive-logs-${WANDB_RUN_GROUP}-${ID}/" --ckpt_dir=".cache/${WANDB_RUN_GROUP}" --linear_monitor --temporal_jitter_range=${i} --preload_dataset --resolution ${j} --wandb; 
	done;
