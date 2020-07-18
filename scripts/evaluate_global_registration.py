"""
A collection of unrefactored functions.
"""
import os
import sys

import argparse
import logging
import open3d as o3d
import numpy as np

from lib.timer import Timer, AverageMeter

from util.misc import extract_features

from model import load_model
from util.file import ensure_dir, get_folder_list, get_file_list
from util.trajectory import read_trajectory, write_trajectory
from util.pointcloud import make_open3d_point_cloud, evaluate_feature_3dmatch, make_open3d_feature
from scripts.benchmark_util import do_single_pair_matching, gen_matching_pair, gather_results
from util.visualization import get_colored_point_cloud_feature

import torch
import copy
import ruamel.yaml as yaml
import math

import MinkowskiEngine as ME

_EPS = np.finfo(float).eps * 4.0

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

testcases_dict = {
    'cheezit': [
        'cheezit_easy_00_00', 'cheezit_easy_00_01','cheezit_easy_00_02','cheezit_easy_00_03','cheezit_easy_00_04',
        'cheezit_easy_00_05','cheezit_easy_00_06','cheezit_easy_00_07','cheezit_easy_00_08','cheezit_easy_00_09',
        'cheezit_easy_00_10', 'cheezit_easy_00_11', 'cheezit_easy_00_12', 'cheezit_easy_00_13', 'cheezit_easy_00_14',
        'cheezit_hard_00_00', 'cheezit_hard_00_01','cheezit_hard_00_02','cheezit_hard_00_03','cheezit_hard_00_04',
        'cheezit_hard_00_05','cheezit_hard_00_06','cheezit_hard_00_07','cheezit_hard_00_08','cheezit_hard_00_09',
        'cheezit_hard_00_10','cheezit_hard_00_11','cheezit_hard_00_12','cheezit_hard_00_13','cheezit_hard_00_14'
    ],
    'bleach': [
        'bleach_easy_00_00', 'bleach_easy_00_01', 'bleach_easy_00_02', 'bleach_easy_00_03', 'bleach_easy_00_04',
        'bleach_easy_00_05', 'bleach_easy_00_06', 'bleach_easy_00_07', 'bleach_easy_00_08', 'bleach_easy_00_09',
        'bleach_hard_00_00', 'bleach_hard_00_01', 'bleach_hard_00_02', 'bleach_hard_00_03', 'bleach_hard_00_04',
        'bleach_hard_00_05', 'bleach_hard_00_06', 'bleach_hard_00_07', 'bleach_hard_00_08', 'bleach_hard_00_09'
    ]
}

def quaternion_matrix(quaternion, trans):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], trans[0]],
        [q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], trans[1]],
        [q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], trans[2]],
        [0, 0, 0, 1]])


def load_pose(path):
    """
    :param path: Path to a file with poses.
    :return: List of the loaded poses.
    """
    trans_mat = []
    with open(path, 'r') as f:
        res = yaml.load(f, Loader=yaml.CLoader)
        if res is None:
            return []

        for est in res:
            q = np.array(est['rotation'])
            trans_mat = quaternion_matrix(q, est['translation'])

    return trans_mat


def voxel_evaluation(pred_cloud, gt_cloud, voxel_size=0.005):
    """
    evaluation of two pointclouds using voxelgrid

    params:
        pred_cloud: o3d.geometry.PointCloud
        gt_cloud: o3d.geometry.PointCloud
        voxel_size: double
    return:
        dict: {'iou': float, 'precision': float, 'recall': float}
    """
    pred_cloud = copy.deepcopy(pred_cloud)
    gt_cloud = copy.deepcopy(gt_cloud)

    min_bound = np.minimum(*[cloud.get_min_bound() for cloud in [pred_cloud, gt_cloud]])
    max_bound = np.maximum(*[cloud.get_max_bound() for cloud in [pred_cloud, gt_cloud]])
    cube_size = ((max_bound - min_bound) / voxel_size).astype(int) + 1

    pred_cloud.translate(-min_bound)
    gt_cloud.translate(-min_bound)

    pred_cube = np.zeros(cube_size, dtype=bool)
    gt_cube = np.zeros(cube_size, dtype=bool)

    pred_indices = (np.asarray(pred_cloud.points) / voxel_size).astype(int)
    gt_indices = (np.asarray(gt_cloud.points) / voxel_size).astype(int)

    pred_cube[pred_indices[:, 0], pred_indices[:, 1],
              pred_indices[:, 2]] = True
    gt_cube[gt_indices[:, 0], gt_indices[:, 1], gt_indices[:, 2]] = True

    intersection = (pred_cube & gt_cube).sum()
    union = (pred_cube | gt_cube).sum()

    precision = intersection.sum() / pred_cube.sum()
    recall = intersection.sum() / gt_cube.sum()
    iou = intersection / union

    return {'iou': iou, 'precision': precision, 'recall': recall}


def distance_evaluation(pred_cloud, gt_cloud, dist_thresh=0.005):
    """
    evaluation of two point clouds using closest distance metric

    params:
        pred_cloud: o3d.geometry.PointCloud
        gt_cloud: o3d.geometry.PointCloud
        dist_thresh: double
    return:
        dict: {'precision': float, 'recall': float}
    """
    pred_to_gt_dist = pred_cloud.compute_point_cloud_distance(gt_cloud)
    gt_to_pred_dist = gt_cloud.compute_point_cloud_distance(pred_cloud)
    precision = (np.asarray(pred_to_gt_dist) < dist_thresh).sum() / len(np.asarray(pred_to_gt_dist))
    recall = (np.asarray(gt_to_pred_dist) < dist_thresh).sum() / len(np.asarray(gt_to_pred_dist))
    return precision, recall


def eval_shape(pred_cloud_old, gt_cloud_old, gt_pose, dist_thresh=0.01):
    pred_cloud = copy.deepcopy(pred_cloud_old)
    gt_cloud = copy.deepcopy(gt_cloud_old)
    gt_cloud.transform(gt_pose)
    return distance_evaluation(pred_cloud, gt_cloud, dist_thresh=dist_thresh)


def update_seen_cloud(scene_seen_cloud, scene_unseen_cloud, model_seen_cloud):
    dist_threshold_seen = 0.005
    dist_threshold_unseen = 0.01

    seen_tree = o3d.geometry.KDTreeFlann(scene_seen_cloud)
    unseen_tree = o3d.geometry.KDTreeFlann(scene_unseen_cloud)

    for pt in model_seen_cloud.points:
        [k1, idx1, dist1] = seen_tree.search_knn_vector_3d(pt, 1)
        if dist1[0] < dist_threshold_seen * dist_threshold_seen:
            continue

        [k2, idx2, dist2] = unseen_tree.search_knn_vector_3d(pt, 1)
        if dist2[0] < dist_threshold_unseen * dist_threshold_unseen:
            scene_seen_cloud.points.append(pt)


def update_unseen_cloud(model_seen_cloud, model_unseen_cloud, scene_unseen_cloud):
    dist_threshold_seen = 0.005
    dist_threshold_unseen = 0.005

    seen_tree = o3d.geometry.KDTreeFlann(model_seen_cloud)
    unseen_tree = o3d.geometry.KDTreeFlann(model_unseen_cloud)

    empty_indices = []

    for idx, pt in enumerate(scene_unseen_cloud.points):
        [k1, idx1, dist1] = seen_tree.search_knn_vector_3d(pt, 1)
        if dist1[0] < dist_threshold_seen * dist_threshold_seen:
            continue

        [k2, idx2, dist2] = unseen_tree.search_knn_vector_3d(pt, 1)
        if dist2[0] < dist_threshold_unseen * dist_threshold_unseen:
            continue

        empty_indices.append(idx)

    if o3d.__version__ == '0.10.0.0':
        updated_scene_unseen_cloud = scene_unseen_cloud.select_by_index(empty_indices, True)
    else:
        updated_scene_unseen_cloud = scene_unseen_cloud.select_down_sample(empty_indices, True)
    return updated_scene_unseen_cloud


def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.0
    print(":: Apply fast global registration with distance threshold %.3f" % distance_threshold)
    result = o3d.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold)
    )
    return result


def execute_ransac_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.0
    ransac_result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        criteria=o3d.registration.RANSACConvergenceCriteria(200000, 10000),
        # checkers=[
        #     o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        #     o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        # ],
    )
    return ransac_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('--voxel_size', default=0.005, type=float, help='voxel size to preprocess point cloud')
    parser.add_argument('--data_path', default='./eval_registration', type=str)
    parser.add_argument('--object_model', default='bleach', type=str)
    parser.add_argument('--output_dir', default='.', type=str)
    args = parser.parse_args()

    data_path = args.data_path
    assert args.checkpoint is not None
    os.makedirs(os.path.join(args.output_dir, args.object_model), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_txt = open(os.path.join(args.output_dir, "result_" + args.object_model + '.txt'), 'w')

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
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

    mean_precision = []
    mean_recall = []

    testcases = testcases_dict[args.object_model]

    object_model_path = os.path.join(data_path, "models", args.object_model + ".ply")
    object_model_voxelized = o3d.io.read_point_cloud(object_model_path)
    object_model_voxelized.paint_uniform_color(np.array([1, 1, 0]))

    for i in range(0, len(testcases)):
        for j in range(0, len(testcases)):
            if i == j: continue

            pcd1_seen = o3d.io.read_point_cloud(os.path.join(data_path, testcases[j], 'final_seen_cloud.ply'))
            pcd1_unseen = o3d.io.read_point_cloud(os.path.join(data_path, testcases[j], 'final_unseen_cloud.ply'))

            pcd2_seen = o3d.io.read_point_cloud(os.path.join(data_path, testcases[i], 'init_seen_cloud.ply'))
            pcd2_unseen = o3d.io.read_point_cloud(os.path.join(data_path, testcases[i], 'init_unseen_cloud.ply'))

            with torch.no_grad():
                xyz_down1, feature1 = extract_features(
                    model,
                    xyz=np.array(pcd1_seen.points),
                    rgb=None,
                    normal=None,
                    voxel_size=args.voxel_size,
                    device=device,
                    skip_check=True)

                xyz_down2, feature2 = extract_features(
                    model,
                    xyz=np.array(pcd2_seen.points),
                    rgb=None,
                    normal=None,
                    voxel_size=args.voxel_size,
                    device=device,
                    skip_check=True)

                pcd_down1 = make_open3d_point_cloud(xyz_down1)
                pcd_down2 = make_open3d_point_cloud(xyz_down2)

                F1 = feature1.clone().detach()
                F2 = feature2.clone().detach()

                feat1 = make_open3d_feature(F1, 32, F1.shape[0])
                feat2 = make_open3d_feature(F2, 32, F2.shape[0])

                ##### Try RANSAC
                ransac_result = execute_ransac_registration(pcd_down1, pcd_down2, feat1, feat2, args.voxel_size)

                ##### Try FGR
                # ransac_result = execute_fast_global_registration(pcd_down1, pcd_down2, feat1, feat2, args.voxel_size)

                pcd2_seen_temp = copy.deepcopy(pcd2_seen).paint_uniform_color(np.array([1, 1, 0]))
                pcd2_unseen_temp = copy.deepcopy(pcd2_unseen).paint_uniform_color(np.array([0, 0, 1]))

                pcd1_seen.transform(ransac_result.transformation)
                pcd1_unseen.transform(ransac_result.transformation)

                combined_pcd2 = pcd2_seen_temp + pcd2_unseen_temp
                combined_cloud = pcd1_seen + pcd1_unseen + combined_pcd2
                output_cloud_path = os.path.join(args.output_dir, args.object_model,
                                                 testcases[i] + '_' + testcases[j] + '.ply')
                o3d.io.write_point_cloud(output_cloud_path, combined_cloud)

                gt_pose_filepath = os.path.join(data_path, testcases[i], 'gt.yml')
                gt_pose = load_pose(gt_pose_filepath)

                update_seen_cloud(pcd2_seen, pcd2_unseen, pcd1_seen)
                pcd2_unseen = update_unseen_cloud(pcd1_seen, pcd1_unseen, pcd2_unseen)

                pcd2_complete_init = pcd2_seen_temp + pcd2_unseen_temp
                pcd2_complete_final = pcd2_seen + pcd2_unseen

                object_model_voxelized = o3d.io.read_point_cloud(object_model_path)
                precision, recall = eval_shape(pcd2_complete_final, object_model_voxelized, gt_pose)

                transformed_obj_model = copy.deepcopy(object_model_voxelized)
                transformed_obj_model.transform(gt_pose)

                txt = '%s %s %f %f\n' % (testcases[i], testcases[j], precision, recall)
                print(txt, end='', flush=True)
                output_txt.write(txt)

                mean_precision.append(precision)
                mean_recall.append(recall)

    output_txt.close()
    print('mean precision: ', np.mean(mean_precision))
    print('mean recall: ', np.mean(mean_recall))
