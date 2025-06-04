#!/bin/bash

DEVICE=$1
MODE=$2  # patch 或 seg

CMD="CUDA_VISIBLE_DEVICES=${DEVICE} \
    python3 ./src/train.py \
    --train_file './dataset/CASIA-Iris-Lamp/train_pairs.txt' \
    --epochs 100 \
    --lr 0.01 \
    --milestones 30 55 75 90"

if [ "$MODE" = "patch" ]; then
    CMD="$CMD --patch"
elif [ "$MODE" = "seg" ]; then
    CMD="$CMD --segmentation"
fi

eval "$CMD"