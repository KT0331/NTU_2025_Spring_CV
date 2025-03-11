#!/bin/bash

run_mode=${1:-'t'}

if [ "$run_mode" = "t" ]; then
    python3 eval.py --image_path ./testdata/ex.png --gt_bf_path ./testdata/ex_gt_bf.png --gt_jbf_path ./testdata/ex_gt_jbf.png
fi

if [ "$run_mode" = "p" ]; then
    python3 main.py --image_path ./testdata/$2.png --setting_path ./testdata/$2_setting.txt
fi
