#!/bin/bash

# for k in 0 1 2 3;
#     do for i in 0 50;
# 	    do export CUDA_VISIBLE_DEVICES=0,1 && export WANDB_RUN_GROUP=barlow_stream51-cifar_byolaugs_time_jittering_deterministic${i} && python main.py --config_file="configs/barlow_stream51.yaml" --data_dir="../stream_data" --log_dir="../logs/stream51-contrastive-logs-${WANDB_RUN_GROUP}-${ID}/" --ckpt_dir=".cache/${WANDB_RUN_GROUP}" --linear_monitor --temporal_jitter_range=${i} --preload_dataset --download --wandb;
#     done;
# done;

export CUDA_VISIBLE_DEVICES=0 && export WANDB_RUN_GROUP=byol_cifar_sanity_sgd && python main.py --config_file="configs/byol_cifar.yaml" --data_dir="../stream_data" --log_dir="../logs/cifar-contrastive-logs-${WANDB_RUN_GROUP}-${ID}/" --ckpt_dir=".cache/${WANDB_RUN_GROUP}" --linear_monitor --temporal_jitter_range=0 --preload_dataset --download --small_dataset
