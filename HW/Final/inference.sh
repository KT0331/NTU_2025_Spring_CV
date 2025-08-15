#!/bin/bash

DEVICE=$1
MODE=$2  # 'patch' or 'seg' or keep empty

CMD="CUDA_VISIBLE_DEVICES=${DEVICE} \
    python3 ./src/run.py \
    --input './input_list/list_CASIA-Iris-Lamp.txt' \
    --checkpoint_file_path './src/ckpt_weights_Lamp'"

if [ "$MODE" = "patch" ]; then
    CMD="$CMD --patch \
        --output './src/score/lamp_patch.txt'"
elif [ "$MODE" = "seg" ]; then
    CMD="$CMD --segmentation \
        --output './src/score/lamp_segmentation.txt'"
else
    CMD="$CMD --output './src/score/lamp_origin.txt'"
fi

eval "$CMD"

if [ "$MODE" = "patch" ]; then
    python3 ./src/eval.py --input './src/score/lamp_patch.txt'
elif [ "$MODE" = "seg" ]; then
    python3 ./src/eval.py --input './src/score/lamp_segmentation.txt'
else
    python3 ./src/eval.py --input './src/score/lamp_origin.txt'
fi