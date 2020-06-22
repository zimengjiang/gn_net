#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python run.py \
    --dataset_name 'cmu' \
    --dataset_root '/local-scratch/fuyang/dad/' \
    --save_root '/local-scratch/fuyang/dad/gn_net/ckpt_cmu/park' \
    --dataset_image_folder 'images' \
    --pair_info_folder 'correspondence/park' \
    --query_folder 'query' \
    --gn_loss_lamda '0.5' \
    --contrastive_lamda '1' \
    --margin_pos 0.05 \
    --margin_neg 1 \
    --scale 4 \
    --total_epochs 200 \
    --lr 1e-3 \
    --schedule_lr_frequency 1 \
    --schedule_lr_fraction 0.85 \
    --weight_decay 0.1 \
    --validation_frequency 5 \
    --notes 'img scale 4 channel 128 on cmu park slices' \
    --resume_checkpoint '/local-scratch/fuyang/dad/gn_net/ckpt_cmu/park/0_checkpoint.pth.tar'
# --num_matches '4000' 


