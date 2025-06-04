python ./training_data_generator/data_generator.py --root './dataset/origin/CASIA-Iris-Lamp' \
       --num_pairs 40000 \
       --val_ratio 0.00 \
       --exclude_range 0,199\
       --out_dir './dataset/CASIA-Iris-Lamp'