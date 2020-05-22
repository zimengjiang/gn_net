#!/bin/bash

python run.py \
--dataset_root '/local/home/lixxue/Downloads/gn_net_data' \
--dataset_name 'cmu' \
--dataset_image_folder 'images' \
--pair_info_folder 'correspondence' \
--query_folder 'query' \
--scale 4 \
--total_epochs 100 \
--save_root '/local/home/lixxue/gnnet/checkpoint_cmu/ckp1' \
--gn_loss_lamda '0.01' \
--contrastive_lamda '1' \
--lr 1e-5 \
--schedule_lr_fraction '0.1' \
--margin '0.5' \
--weight_decay 0.01 \
--validation_frequency 1  
# --num_matches '4000' 
# --resume_checkpoint '/Users/zimengjiang/code/3dv/ours/S2DHM/checkpoints/gnnet/25_model_best.pth.tar'
# --save_root '/Users/zimengjiang/code/3dv/ours/checkpoint' \ 
# --dataset_root '/Users/zimengjiang/code/3dv/public_data' \


