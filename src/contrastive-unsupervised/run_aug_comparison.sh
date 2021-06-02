#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1 && export WANDB_RUN_GROUP=barlow_stream51_augmentationcomparison && python main.py --config_file="configs/barlow_stream51.yaml" --data_dir="../stream_data" --log_dir="../logs/cifar-contrastive-logs-${WANDB_RUN_GROUP}-${ID}/" --ckpt_dir=".cache/${WANDB_RUN_GROUP}" --linear_monitor --temporal_jitter_range=50 --preload_dataset --download --single_aug Nothing --bbox_crop --wandb;

for aug in RandomResizedCrop RandomHorizontalFlip ColorJitter RandomGrayscale Solarization;
    do export CUDA_VISIBLE_DEVICES=0,1 && export WANDB_RUN_GROUP=barlow_stream51_augmentationcomparison && python main.py --config_file="configs/barlow_stream51.yaml" --data_dir="../stream_data" --log_dir="../logs/cifar-contrastive-logs-${WANDB_RUN_GROUP}-${ID}/" --ckpt_dir=".cache/${WANDB_RUN_GROUP}" --linear_monitor --temporal_jitter_range=0 --preload_dataset --download --single_aug ${aug} --bbox_crop --wandb;
done;
