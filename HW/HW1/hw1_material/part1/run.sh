#!/bin/bash

run_mode=${1:-'t'}

if [ "$run_mode" = "t" ]; then
    python3 eval.py --threshold $2 --image_path ./testdata/1.png --gt_path ./testdata/1_gt.npy
fi

if [ "$run_mode" = "p" ]; then
    python3 main.py --threshold $2 --image_path ./testdata/$3.png --plot_mode $4
fi
