import copy
import open3d as o3d
import numpy as np
import torch

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def get_color_map(x):
  colours = plt.cm.Spectral(x)
  return colours[:, :3]


def mesh_sphere(pcd, voxel_size, sphere_size=0.6):
  # Create a mesh sphere
  spheres = o3d.geometry.TriangleMesh()
  s = o3d.geometry.TriangleMesh.create_sphere(radius=voxel_size * sphere_size)
  s.compute_vertex_normals()

  for i, p in enumerate(pcd.points):
    si = copy.deepcopy(s)
    trans = np.identity(4)
    trans[:3, 3] = p
    si.transform(trans)
    si.paint_uniform_color(pcd.colors[i])
    spheres += si
  return spheres


def get_colored_point_cloud_feature(pcd, feature, voxel_size):
  tsne_results = embed_tsne(feature)

  color = get_color_map(tsne_results)
  pcd.colors = o3d.utility.Vector3dVector(color)
  spheres = mesh_sphere(pcd, voxel_size)

  return spheres


def embed_tsne(data):
  """
  N x D np.array data
  """
  tsne = TSNE(n_components=1, verbose=0, perplexity=40, n_iter=300, random_state=0)
  tsne_results = tsne.fit_transform(data)
  tsne_results = np.squeeze(tsne_results)
  tsne_min = np.min(tsne_results)
  tsne_max = np.max(tsne_results)
  return (tsne_results - tsne_min) / (tsne_max - tsne_min)


def visualize_correspondence(pcd1, pcd2, feat1, feat2):
  """
  pcd1, pcd2: o3d.geometry.PointCloud
  feat1, feat2: torch.tensor of size (n_points, n_dim)
  """

  assert len(pcd1.points) == len(feat1)
  assert len(pcd2.points) == len(feat2)

  if feat1.device != torch.device('cpu'):
    feat1 = feat1.detach().cpu()
  if feat2.device != torch.device('cpu'):
    feat2 = feat2.detach().cpu()

  pcd1_vis = copy.deepcopy(pcd1)
  pcd2_vis = copy.deepcopy(pcd2)

  # colorize pcd2 using tsne and spectual color
  tsne_results = embed_tsne(feat2)
  color = get_color_map(tsne_results)
  pcd2_vis.colors = o3d.utility.Vector3dVector(color)

  feat1 = feat1.numpy().T
  feat2 = feat2.numpy().T
  pcd1_vis.colors = o3d.utility.Vector3dVector(np.zeros((feat1.shape[1], 3)))

  feat_kdtree = o3d.geometry.KDTreeFlann(feat2)
  for i in range(feat1.shape[1]):
    k, indices, _ = feat_kdtree.search_knn_vector_xd(feat1[:, i], knn=1)
    idx = indices[0]  # only choose the closest one, i.e. k = 1
    pcd1_vis.colors[i] = pcd2_vis.colors[idx]

  return pcd1_vis, pcd2_vis
