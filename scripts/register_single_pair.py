"""
A collection of unrefactored functions.
"""
import os
import sys
import numpy as np
import argparse
import copy
import logging

import open3d as o3d
import torch

from model import load_model
from util.misc import extract_features
from util.file import ensure_dir
from util.pointcloud import make_open3d_point_cloud, evaluate_feature_3dmatch, make_open3d_feature
from util.visualization import get_colored_point_cloud_feature


ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source1', default=None, type=str, help='path to the source point cloud')
    parser.add_argument('--source2', default=None, type=str, help='path to the target point cloud')
    parser.add_argument('--model', default=None, type=str, help='path to the checkpoint')
    parser.add_argument('--voxel_size', default=0.005, type=float, help='voxel size (m) to preprocess point cloud')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert args.model is not None
    assert args.source1 is not None

    checkpoint = torch.load(args.model, map_location='cpu')
    config = checkpoint['config']

    num_feats = 1
    Model = load_model(config.model)
    model = Model(
        num_feats,
        config.model_n_out,
        bn_momentum=0.05,
        normalize_feature=config.normalize_feature,
        conv1_kernel_size=config.conv1_kernel_size,
        D=3)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    model = model.to(device)
    pcd1 = o3d.io.read_point_cloud(args.source1)
    pcd1.transform([
        [1, 0, 0, 0.3],
        [0, 1, 0, 0.4],
        [0, 0, 1, -0.3],
        [0, 0, 0, 1]
    ])
    pcd2 = o3d.io.read_point_cloud(args.source2)

    o3d.visualization.draw_geometries([pcd1, pcd2])

    with torch.no_grad():
        xyz_down1, feature1 = extract_features(
            model,
            xyz=np.array(pcd1.points),
            rgb=None,
            normal=None,
            voxel_size=args.voxel_size,
            device=device,
            skip_check=True
        )

        xyz_down2, feature2 = extract_features(
            model,
            xyz=np.array(pcd2.points),
            rgb=None,
            normal=None,
            voxel_size=args.voxel_size,
            device=device,
            skip_check=True
        )

        # Visualize T-SNE
        # col_pcd1 = get_colored_point_cloud_feature(pcd1, feature1.cpu(), args.voxel_size)
        # col_pcd2 = get_colored_point_cloud_feature(pcd2, feature2.cpu(), args.voxel_size)
        # o3d.visualization.draw_geometries([col_pcd1, col_pcd2])

        pcd_down1 = make_open3d_point_cloud(xyz_down1)
        pcd_down2 = make_open3d_point_cloud(xyz_down2)

        F1 = feature1.clone().detach()
        F2 = feature2.clone().detach()

        feat1 = make_open3d_feature(F1, 32, F1.shape[0])
        feat2 = make_open3d_feature(F2, 32, F2.shape[0])

        ##### Try RANSAC
        distance_threshold = args.voxel_size * 1.0
        ransac_result = o3d.registration.registration_ransac_based_on_feature_matching(
            pcd_down1, pcd_down2, feat1, feat2,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.registration.TransformationEstimationPointToPoint(
                False),
            ransac_n=4,
            checkers=[
                o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ],
            criteria=o3d.registration.RANSACConvergenceCriteria(4000000, 10000)
        )

        pcd1_transformed = copy.deepcopy(pcd1)
        pcd1_transformed.transform(ransac_result.transformation)

        pcd2.paint_uniform_color([1, 1, 0])
        combined_pcd = pcd1_transformed + pcd2

        o3d.visualization.draw_geometries([combined_pcd])
