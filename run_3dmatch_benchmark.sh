#!/bin/bash

python -m scripts.benchmark_3dmatch \
    --source ./dataset/threedmatch_test \
    --target ./outputs \
    --voxel_size 0.025 \
    --model ./weights/2019-08-19_06-17-41.pth \
    --extract_features \
    --evaluate_feature_match_recall \
    --with_cuda
