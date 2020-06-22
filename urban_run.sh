#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python run.py \
    --dataset_name 'cmu' \
    --dataset_root '/local-scratch/fuyang/dad/' \
    --save_root '/local-scratch/fuyang/dad/gn_net/ckpt_cmu/urban' \
    --dataset_image_folder 'images' \
    --pair_info_folder 'correspondence/urban' \
    --query_folder 'query' \
    --gn_loss_lamda '0.5' \
    --contrastive_lamda '1' \
    --margin_pos 0.05 \
    --margin_neg 1 \
    --scale 4 \
    --total_epochs 200 \
    --schedule_lr_frequency 1 \
    --schedule_lr_fraction 0.85 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --validation_frequency 1 \
    --notes 'img scale 4 channel 128 on cmu urban slices'
#    --schedule_lr_fraction '0.1' \
# --num_matches '4000' 
# --resume_checkpoint '/Users/zimengjiang/code/3dv/ours/S2DHM/checkpoints/gnnet/25_model_best.pth.tar'


