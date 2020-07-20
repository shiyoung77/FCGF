#!/usr/bin/env bash

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=3

CHECK_POINT="experiments/voxel_0.005_mul_3_hit_0.02_NEG_1.4_POS_0.1/best_val_checkpoint.pth"

python -m scripts.evaluate_global_registration \
    --checkpoint $CHECK_POINT \
    --voxel_size 0.01 \
    --data_path ./eval_registration \
    --object_model cheezit \

# python -m scripts.evaluate_global_registration \
#     --checkpoint $CHECK_POINT  \
#     --voxel_size 0.01 \
#     --data_path ./eval_registration \
#     --object_model bleach \
# 
