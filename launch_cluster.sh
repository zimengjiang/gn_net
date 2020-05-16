#!/bin/bash

output="-oo $(pwd)/out.txt"
error="-eo $(pwd)/error.txt"
cores="8"
memory="2000"  # MB per core
scratch="0"  # MB per core
gpus="1"
clock="24:00"  # time limit: 4:00, 24:00, or 120:00
gpu_memory="10240"  # minimum GPU memory
# gpu_memory="15360"  # minimum GPU memory
warn="-wt 10 -wa INT"  # interrupt signal 10 min before timeout

cmd="bsub
    -n $cores
    -W $clock $output
    $warn
    -R 'select[gpu_model0==TeslaV100_SXM2_32GB] rusage[mem=$memory,scratch=$scratch,ngpus_excl_p=$gpus]'
    $*"
echo $cmd
eval $cmd

# https://scicomp.ethz.ch/wiki/Getting_started_with_GPUs
