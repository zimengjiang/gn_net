#!/bin/bash

python run.py \
--dataset_name 'robotcar' \
--dataset_root '/cluster/work/riner/users/PLR-2020/lechen/lucky' \
--save_root '/cluster/work/riner/users/PLR-2020/lechen/lucky/new_ckpt/025_weighted_night_finetune_vgg16_s2d_lr1e-7_gnloss_gnlambda05_marginpos01neg1_match1024_fraction085_wd005' \
--dataset_image_folder 'images' \
--pair_info_folder 'correspondence' \
--query_folder 'query' \
--num_workers 4 \
--gn_loss_lamda '0.5' \
--contrastive_lamda '1' \
--margin_pos 0.1 \
--margin_neg 1 \
--e1_lamda 1 \
--e2_lamda 0.28571428571 \
--scale 1 \
--total_epochs 2 \
--schedule_lr_frequency 1 \
--schedule_lr_fraction 0.85 \
--lr 1e-7 \
--weight_decay 0.05 \
--validation_frequency 1 \
--log_dir 'logs/025_weighted_night_finetune_vgg16_s2d_lr1e-7_gnloss_gnlambda05_marginpos01neg1_match1024_fraction085_wd005' \
--vgg_checkpoint '/cluster/work/riner/users/PLR-2020/lechen/s2d/S2DHM/checkpoints/robotcar/weights.pth.tar' 
