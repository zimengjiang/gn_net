#!/bin/bash

python run.py \
--dataset_root '/Users/zimengjiang/code/3dv/public_data' \
--dataset_name 'cmu' \
--dataset_image_folder 'images' \
--pair_info_folder 'correspondence' \
--query_folder 'query_all' \
--scale 4 \
--total_epochs 50 \
--save_root '/Users/zimengjiang/code/3dv/ours/checkpoint' \
--gn_loss_lamda '0.00' \
--contrastive_lamda '1' \
--lr 1e-5 \
--validate 'False' \
--schedule_lr_fraction 1\
--margin 0.5
# --num_matches '4000' 
# --resume_checkpoint '/Users/zimengjiang/code/3dv/ours/S2DHM/checkpoints/gnnet/25_model_best.pth.tar'
# --save_root '/Users/zimengjiang/code/3dv/ours/checkpoint' \ 
# --dataset_root '/Users/zimengjiang/code/3dv/public_data' \


