import os
import glob
import numpy as np
import open3d as o3d
from util.pointcloud import compute_overlap_ratio

dataset_path = "datasets/threedmatch"

scenes = []
with open("config/train_3dmatch.txt", 'r') as f:
    line = f.readline().strip()
    while line:
        scenes.append(line)
        line = f.readline().strip()

idx = 0

scene = glob.glob(os.path.join(dataset_path, scenes[0] + "*.txt"))[0]
with open(scene, 'r') as f:
    line = f.readline().strip()
    data1, data2, ratio = line.split()
    scene1 = np.load(os.path.join(dataset_path, data1))
    scene2 = np.load(os.path.join(dataset_path, data2))

    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(scene1['pcd'])
    pcd2.points = o3d.utility.Vector3dVector(scene2['pcd'])
    pcd1.colors = o3d.utility.Vector3dVector(scene1['color'])
    pcd2.colors = o3d.utility.Vector3dVector(scene2['color'])

    pcd1.paint_uniform_color([1, 1, 0])
    pcd2.paint_uniform_color([0, 0, 1])

    trans = np.identity(4)
    print("ratio", ratio)
    print("overlap_ratio", compute_overlap_ratio(pcd1, pcd2, trans, 0.1))

    o3d.visualization.draw_geometries([pcd1, pcd2])
