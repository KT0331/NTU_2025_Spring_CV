#!/bin/bash

DEVICE=$1
MODE=$2  # patch 或 seg

CMD="CUDA_VISIBLE_DEVICES=${DEVICE} \
    python3 ./src/run.py \
    --input './input_list/list_CASIA-Iris-Lamp.txt' \
    --checkpoint_file_path './src/ckpt_weights_Lamp'"

if [ "$MODE" = "patch" ]; then
    CMD="$CMD --patch \
        --output './src/lamp_patch.txt'"
elif [ "$MODE" = "seg" ]; then
    CMD="$CMD --segmentation \
        --output './src/lamp_segmentation.txt'"
else
    CMD="$CMD --output './src/lamp_origin.txt'"
fi

eval "$CMD"

if [ "$MODE" = "patch" ]; then
    python3 ./src/eval.py --input './src/lamp_patch.txt'
elif [ "$MODE" = "seg" ]; then
    python3 ./src/eval.py --input './src/lamp_segmentation.txt'
else
    python3 ./src/eval.py --input './src/lamp_origin.txt'
fi



# CUDA_VISIBLE_DEVICES=$1 \
# python3 ./src/run.py \
# --input './input_list/list_CASIA-Iris-Lamp.txt' \
# --output './src/lamp_origin.txt' \
# --checkpoint_file_path './src/custom/ckpt_weights_Lamp'

# CUDA_VISIBLE_DEVICES=$1 \
# python3 ./src/run.py \
# --input './input_list/list_CASIA-Iris-Lamp.txt' \
# --output './src/lamp_patch.txt' \
# --checkpoint_file_path './src/custom/ckpt_weights_Lamp' \
# --patch

# CUDA_VISIBLE_DEVICES=$1 \
# python3 ./src/run.py \
# --input './input_list/list_CASIA-Iris-Lamp.txt' \
# --checkpoint_file_path './src/custom/ckpt_weights_Lamp' \
# --segmentation

# python3 ./src/eval.py --input './src/score/lamp_origin.txt'
# python3 ./src/eval.py --input './src/score/lamp_patch.txt'
# python3 ./src/eval.py --input './src/score/lamp_segmentation.txt'