#!/bin/bash

python run_vgg.py \
--dataset_root '/Users/zimengjiang/code/3dv/public_data' \
--dataset_name 'robotcar' \
--dataset_image_folder 'images' \
--pair_info_folder 'correspondence' \
--query_folder 'query_all' \
--total_epochs 100 \
--save_root '/Users/zimengjiang/code/3dv/ours/checkpoint' \
--gn_loss_lamda '0.5' \
--contrastive_lamda '1' \
--margin_pos 0.05 \
--margin_neg 1 \
--scale 2 \
--lr 1e-6 \
--schedule_lr_frequency 1 \
--schedule_lr_fraction 0.85 \
--weight_decay 0.1 \
--validation_frequency 1 \
--vgg_checkpoint  '/Users/zimengjiang/code/3dv/ours/compare/S2DHM-master/checkpoints/robotcar/weights.pth.tar'
# --num_matches '4000' 
# --resume_checkpoint '/Users/zimengjiang/code/3dv/ours/S2DHM/checkpoints/gnnet/25_model_best.pth.tar'
# --save_root '/Users/zimengjiang/code/3dv/ours/checkpoint' \ 
# --dataset_root '/Users/zimengjiang/code/3dv/public_data' \


