#!/usr/bin/env bash

python -m scripts.register_single_pair \
    --source1 eval_registration/cheezit_hard_00_12/final_seen_cloud.ply \
    --source2 eval_registration/cheezit_hard_00_14/init_seen_cloud.ply \
    --checkpoint ./weights/2019-08-19_06-17-41.pth \
    --voxel_size 0.01
