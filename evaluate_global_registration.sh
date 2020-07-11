#!/usr/bin/env bash

python -m scripts.evaluate_global_registration \
    --checkpoint experiments/voxel_size_0.005/best_val_checkpoint.pth \
    --voxel_size 0.005 \
    --data_path ./eval_registration \
    --object_model cheezit \
    --output_dir outputs

python -m scripts.evaluate_global_registration \
    --checkpoint experiments/voxel_size_0.005/best_val_checkpoint.pth \
    --voxel_size 0.005 \
    --data_path ./eval_registration \
    --object_model bleach \
    --output_dir outputs
