#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
python opencood/tools/train.py -y opencood/hypes_yaml/v2xsim2/where2comm_multiclass_config.yaml

#> core.txt 2>&1 &
