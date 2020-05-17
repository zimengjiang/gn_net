#!/bin/bash

python run.py \
--dataset_root '/local/home/lixxue/gnnet/gn_net_data_tiny' \
--dataset_name 'cmu' \
--dataset_image_folder 'images' \
--pair_info_folder 'correspondence' \
--query_folder 'query' \
--scale 2 \
--total_epochs 50 \
--save_root '/local/home/lixxue/gnnet/checkpoint' \
--gn_loss_lamda '0.002' \
--contrastive_lamda '100' \
--lr 1e-6
# --num_matches '4000' 
# --resume_checkpoint '/Users/zimengjiang/code/3dv/ours/S2DHM/checkpoints/gnnet/25_model_best.pth.tar'
