#!/bin/bash

PATCH_IN="./dataset/origin/CASIA-Iris-Lamp"
PATCH_OUT="./dataset/patch/CASIA-Iris-Lamp"

if [ ! -d "$PATCH_IN" ]; then
    echo "***************************"
    echo "*** Cannot Find Dataset ***"
    echo "***************************"
    exit 1
fi

echo "=== Generate eye patches start ==="
CUDA_VISIBLE_DEVICES=$1 python ./dataset/patch/patch_gen/preprocess_crop_eye.py \
                                --root "$PATCH_IN" --out "$PATCH_OUT"
echo "=== Generate eye patches end ==="