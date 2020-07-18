import os
import numpy as np
import open3d as o3d
from util.pointcloud import compute_overlap_ratio

scenes = []
with open("./config/train_icra21.txt", 'r') as f:
    line = f.readline().strip()
    while line:
        scenes.append(line)
        line = f.readline().strip()

dataset_path = "datasets/icra21_dataset/data"
idx = 13
scene = os.path.join(dataset_path, scenes[idx] + "-all.txt")
print(scene)

def exit_callback(vis):
    print('exit')
    exit(0)

# vis = o3d.visualization.Visualizer()
# vis.create_window()

with open(scene, 'r') as f:
    line = f.readline().strip()
    while line:
        data1, data2, ratio = line.split()
        scene1 = np.load(os.path.join(dataset_path, data1))
        scene2 = np.load(os.path.join(dataset_path, data2))
        print(data1, data2)

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
        print("overlap_ratio", compute_overlap_ratio(pcd1, pcd2, trans, 0.01))

        o3d.visualization.draw_geometries_with_key_callbacks([pcd1, pcd2], {ord('X'): exit_callback})
        # vis.add_geometry(pcd1)
        # vis.add_geometry(pcd2)
        # vis.poll_events()
        # vis.update_renderer()
        # vis.clear_geometries()

        line = f.readline().strip()
