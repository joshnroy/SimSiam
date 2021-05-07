#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,2 export WANDB_RUN_GROUP=ucf101_supervisedclassifier && python -m cProfile -o test.prof supervised_classifier.py --config_file="configs/simsiam_ucf101.yaml" --data_dir="../ucfimages64x64" --log_dir="../logs/ucf101-supervised-logs-${WANDB_RUN_GROUP}-${ID}/" --ckpt_dir=".cache/${WANDB_RUN_GROUP}" --wandb
