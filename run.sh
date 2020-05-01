#!/bin/bash

python run.py \
--dataset_root '/cluster/work/riner/users/PLR-2020/lechen/gn_net/gn_net_data' \
--dataset_name 'cmu' \
--dataset_image_folder 'images' \
--pair_info_folder 'correspondence' \
--query_folder 'query' \
--scale 2
