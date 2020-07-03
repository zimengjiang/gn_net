#!/bin/bash

python run.py \
--dataset_name 'robotcar' \
--dataset_root '/home/lechen/gnnet/robotcar_data' \
--save_root '/home/lechen/gnnet/checkpoint/rbc_ckpt' \
--dataset_image_folder 'images' \
--pair_info_folder 'correspondence' \
--query_folder 'query' \
--gn_loss_lamda '0.1' \
--contrastive_lamda '1' \
--margin_pos 0.1 \
--margin_neg 1 \
--e1_lamda 1 \
--e2_lamda 1 \
--scale 2 \
--total_epochs 30 \
--schedule_lr_frequency 1 \
--schedule_lr_fraction 0.85 \
--lr 1e-4 \
--weight_decay 0.05 \
--validation_frequency 1 \
--notes 'img scale 4 channel 128 on cmu park slices' \
--log_dir 'logs/vgg/rbc' \
--vgg_checkpoint '/home/lechen/gnnet/evaluate/backup/S2DHM_2/checkpoints/robotcar/weights.pth.tar'

