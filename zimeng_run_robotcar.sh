#!/bin/bash

python run.py \
--dataset_root '/Users/zimengjiang/code/3dv/public_data' \
--dataset_name 'robotcar' \
--dataset_image_folder 'images' \
--pair_info_folder 'correspondence' \
--query_folder 'overcas-ref-rear' \
--scale 4 \
--total_epochs 100 \
--save_root '/Users/zimengjiang/code/3dv/ours/checkpoint/robotcar' \
--gn_loss_lamda '0.05' \
--contrastive_lamda '1' \
--lr 1e-4 \
--validate 'True' \
--schedule_lr_fraction '1' \
--margin '0.5' \
--bilinear 'True' \
--nearest 'False' \
--weight_decay 0.001 \
--validation_frequency 1  
# --num_matches '4000' 
# --resume_checkpoint '/Users/zimengjiang/code/3dv/ours/S2DHM/checkpoints/gnnet/25_model_best.pth.tar'
# --save_root '/Users/zimengjiang/code/3dv/ours/checkpoint' \ 
# --dataset_root '/Users/zimengjiang/code/3dv/public_data' \


