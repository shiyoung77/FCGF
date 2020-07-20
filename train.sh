#!/usr/bin/env bash

export OMP_NUM_THREADS=12
export CUDA_VISIBLE_DEVICES=4

HIT_RATIO_THRESH=0.02
SEARCH_MULTIPLIER=2
VOXEL_SIZE=0.01
NEG_THRESH=1.4
POS_THRESH=0.1

OUTPUT_DIR=experiments/voxel_${VOXEL_SIZE}_mul_${SEARCH_MULTIPLIER}_hit_${HIT_RATIO_THRESH}_NEG_${NEG_THRESH}_POS_${POS_THRESH}

mkdir -p $OUTPUT_DIR

# train from scratch
python train.py \
    --out_dir $OUTPUT_DIR \
    --positive_pair_search_voxel_size_multiplier $SEARCH_MULTIPLIER \
    --hit_ratio_thresh $HIT_RATIO_THRESH \
    --voxel_size $VOXEL_SIZE \
    --neg_thresh $NEG_THRESH \
    --pos_thresh $POS_THRESH \
    --lr 0.1 \
    --batch_size 6 \
    --threed_match_dir /freespace/local/datasets/icra21_dataset/data \
    | tee $OUTPUT_DIR/train_log.txt

# start with pretrained weights
# python train.py \
#     --out_dir $OUTPUT_DIR \
#     --positive_pair_search_voxel_size_multiplier $SEARCH_MULTIPLIER \
#     --hit_ratio_thresh $HIT_RATIO_THRESH \
#     --voxel_size $VOXEL_SIZE \
#     --neg_thresh $NEG_THRESH \
#     --pos_thresh $POS_THRESH \
#     --lr 0.01 \
#     --batch_size 6 \
#     --weights ./weights/2019-08-19_06-17-41.pth \
#     --threed_match_dir /freespace/local/datasets/icra21_dataset/data \
#     | tee $OUTPUT_DIR/train_log.txt
