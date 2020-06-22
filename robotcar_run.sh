#!/bin/bash

python run.py \
--dataset_name 'robotcar' \
--dataset_root '/local/home/lixxue/gnnet/robotcar_data_tiny' \
--save_root '/local/home/lixxue/gnnet/ckpt_test' \
--robotcar_weather 'sun' \
--dataset_image_folder 'images' \
--pair_info_folder 'correspondence' \
--query_folder 'query' \
--gn_loss_lamda '0.5' \
--contrastive_lamda '1' \
--margin_pos 0.05 \
--margin_neg 1 \
--scale 4 \
--total_epochs 200 \
--lr 1e-6 \
--schedule_lr_fraction '0.1' \
--weight_decay 0.1
--validation_frequency 1  
--notes 'euclidean pos and neg, monitor feature norm, matches for each feature level:[1024,1024,1024,1024], contrastive loss'

# --num_matches '4000' 
# --resume_checkpoint '/Users/zimengjiang/code/3dv/ours/S2DHM/checkpoints/gnnet/25_model_best.pth.tar'
# --save_root '/Users/zimengjiang/code/3dv/ours/checkpoint' \ 
# --dataset_root '/Users/zimengjiang/code/3dv/public_data' \


