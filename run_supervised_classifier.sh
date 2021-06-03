#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1 export WANDB_RUN_GROUP=stream51_nobbox_supervisedclassifier && python supervised_classifier.py --config_file="configs/simsiam_stream51.yaml" --data_dir="../stream_data" --log_dir="../logs/stream51_nobbox-supervised-logs-${WANDB_RUN_GROUP}-${ID}/" --ckpt_dir=".cache/${WANDB_RUN_GROUP}" --preload_dataset --wandb
