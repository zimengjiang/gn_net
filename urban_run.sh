#!/bin/bash

python run.py \
--dataset_name 'cmu' \
--dataset_root 'data' \
--save_root 'cmu_ckpt/gn/urban' \
--dataset_image_folder 'images' \
--pair_info_folder 'correspondence/urban' \
--query_folder 'query' \
--num_workers 10 \
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
--log_dir 'logs/cmu/gn/urban' \
--vgg_checkpoint '../S2DHM/checkpoints/cmu/weights.pth.tar'
