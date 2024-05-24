import torch
import os
import numpy as np
import open3d as o3d
from pyntcloud import PyntCloud

eps = 1e-6


def set_bounding_box(cameras):
    cameras_positions = torch.stack([camera.position for camera in cameras])
    x_lim_min = torch.min(cameras_positions[:, 0])
    x_lim_max = torch.max(cameras_positions[:, 0])
    y_lim_min = torch.min(cameras_positions[:, 1])
    y_lim_max = torch.max(cameras_positions[:, 1])
    z_lim_min = torch.min(cameras_positions[:, 2])
    z_lim_max = torch.max(cameras_positions[:, 2])
    x_lims = [x_lim_min, x_lim_max]
    y_lims = [y_lim_min, y_lim_max]
    z_lims = [z_lim_min, z_lim_max]
    return x_lims, y_lims, z_lims


def init_voxels(x_lims: list, y_lims: list, z_lims: list, voxel_num: int):
    x_min, x_max = x_lims
    y_min, y_max = y_lims
    z_min, z_max = z_lims

    x_range = torch.linspace(x_min, x_max, voxel_num)
    y_range = torch.linspace(y_min, y_max, voxel_num)
    z_range = torch.linspace(z_min, z_max, voxel_num)

    x_size = (x_max - x_min) / voxel_num
    y_size = (y_max - y_min) / voxel_num
    z_size = (z_max - z_min) / voxel_num

    voxels = torch.meshgrid(x_range, y_range, z_range)
    voxels = torch.stack(voxels, dim=-1)
    voxels = voxels.reshape(-1, 3)
    voxel_size = torch.tensor([x_size, y_size, z_size])
    return voxels, voxel_size


def set_voxels(voxels: torch.Tensor, voxel_sizes: torch.tensor, camera, masks=None):
    R = torch.tensor(camera.R, dtype=torch.float32)
    T = torch.tensor(camera.T, dtype=torch.float32)
    K = torch.tensor(camera.K, dtype=torch.float32)
    # matrix_out = torch.dot(R, T)
    projection_matrix = torch.tensor(camera.P, dtype=torch.float32)
    # volumns_over =voxel_sizes
    voxels_size_over = voxel_sizes
    ones_concat = torch.ones(voxels.shape[0], 1)
    voxel_4d = torch.cat([voxels, ones_concat], dim=-1)
    projection_matrix = projection_matrix.unsqueeze(0).repeat(voxels.shape[0], 1, 1)
    voxel_4d = voxel_4d.unsqueeze(-1)
    voxel_project = torch.bmm(projection_matrix, voxel_4d).squeeze()
    w = voxel_project[:, -1].unsqueeze(-1) + eps
    uv = (voxel_project / w)[:, :2]
    uv = uv.long()
    w_index = torch.logical_and(uv[:, 0] >= 0, uv[:, 0] < camera.W)
    h_index = torch.logical_and(uv[:, 1] >= 0, uv[:, 1] < camera.H)
    index = torch.logical_and(w_index, h_index)
    if masks is not None:
        u_index = uv[:, 0]
        v_index = uv[:, 1]
        masks_index = masks[v_index, u_index] > 0
        index = torch.logical_and(index, masks_index)
    # if (uv[0] >= 0 and uv[0] < camera.W) and (uv[1] >= 0 and uv[1] < camera.H):
    volumns_over = voxels[index]

    return volumns_over, voxels_size_over


def split_to_octree(voxel: torch.Tensor, voxel_size: torch.Tensor):
    voxel_splits = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                voxel_temp = voxel.clone()
                voxel_temp += torch.tensor([(-1) ** i,
                                            (-1) ** j, (-1) ** k]) / 2 * voxel_size
                voxel_splits.append(voxel_temp)
    voxel_splits = torch.stack(voxel_splits)
    voxel_splits = voxel_splits.reshape(-1, 3)
    return voxel_splits


def save_ply(voxel: torch.tensor, path: str):
    xyz = voxel.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    o3d.io.write_point_cloud(path, pcd)


def read_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return points

def load_pts(filename: str):

    cloud = PyntCloud.from_file(filename)
    verts = cloud.xyz
    if 'red' in cloud.points and 'green' in cloud.points and 'blue' in cloud.points:
        r = np.asarray(cloud.points['red'])
        g = np.asarray(cloud.points['green'])
        b = np.asarray(cloud.points['blue'])
        colors = (np.stack([r, g, b], axis=-1) / 255).astype(np.float32)
    elif 'r' in cloud.points and 'g' in cloud.points and 'b' in cloud.points:
        r = np.asarray(cloud.points['r'])
        g = np.asarray(cloud.points['g'])
        b = np.asarray(cloud.points['b'])
        colors = (np.stack([r, g, b], axis=-1) / 255).astype(np.float32)
    else:
        colors = None

    if 'nx' in cloud.points and 'ny' in cloud.points and 'nz' in cloud.points:
        nx = np.asarray(cloud.points['nx'])
        ny = np.asarray(cloud.points['ny'])
        nz = np.asarray(cloud.points['nz'])
        norms = np.stack([nx, ny, nz], axis=-1)
    else:
        norms = None

    # if 'alpha' in cloud.points:
    #     cloud.points['alpha'] = cloud.points['alpha'] / 255

    reserved = ['x', 'y', 'z', 'red', 'green', 'blue', 'r', 'g', 'b', 'nx', 'ny', 'nz']
    scalars = dotdict({k: np.asarray(cloud.points[k])[..., None] for k in cloud.points if k not in reserved})  # one extra dimension at the back added
    return verts, colors, norms, scalars

