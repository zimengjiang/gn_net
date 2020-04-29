#!/bin/bash

python run.py \
--dataset_root '/Users/zimengjiang/code/3dv/public_data' \
--dataset_name 'cmu' \
--dataset_image_folder 'images' \
--pair_info_folder 'correspondence_data' \
--query_folder 'query' \
--scale 4
