#!/bin/bash

python run.py \
--dataset_root '/Users/zimengjiang/code/3dv/public_data' \
--dataset_name 'cmu' \
--dataset_image_folder 'images' \
--pair_info_folder 'correspondence' \
--query_folder 'query_all' \
--scale 4 \
--total_epochs 10 \
--save_root '/Users/zimengjiang/code/3dv/ours/checkpoint' \
--resume_checkpoint '/Users/zimengjiang/code/3dv/ours/S2DHM/checkpoints/gnnet/25_model_best.pth.tar'
