#!/bin/bash

# for k in 0 1 2;
#     do for i in 0 50;
# 	    do export CUDA_VISIBLE_DEVICES=0,1 && export WANDB_RUN_GROUP=simclr_stream51-cifar_time_jittering_deterministic${i} && python main.py --config_file="configs/simclr_stream51.yaml" --data_dir="../stream_data" --log_dir="../logs/stream51-contrastive-logs-${WANDB_RUN_GROUP}-${ID}/" --ckpt_dir=".cache/${WANDB_RUN_GROUP}" --linear_monitor --temporal_jitter_range=${i} --preload_dataset --download --wandb;
#     done;
# done;

for i in 0;
    do export CUDA_VISIBLE_DEVICES=0 && export WANDB_RUN_GROUP=simclr_cifar_sanity && python main.py --config_file="configs/simclr_cifar.yaml" --data_dir="../stream_data" --log_dir="../logs/cifar-contrastive-logs-${WANDB_RUN_GROUP}-${ID}/" --ckpt_dir=".cache/${WANDB_RUN_GROUP}" --linear_monitor --download --wandb;
done;
