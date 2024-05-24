import torch
import cv2
import os
import imageio.v2 as imageio
import numpy as np


from lib.utils.camera.r4k4d_cam_param import CamParams
from lib.utils.ply_utils import *
from lib.config import cfg

class Camera:
    def __init__(self, cam_path_intri, cam_path_extri) -> None:
        self.intrix_file = cv2.FileStorage(cam_path_intri, cv2.FILE_STORAGE_READ)
        self.extrix_file = cv2.FileStorage(cam_path_extri, cv2.FILE_STORAGE_READ)
        self.use_camera_num = cfg.task_arg.camera_use
        self.use_frames = cfg.task_arg.dim_time

        self.cameras_all = []
        cam_names = self.read('names', True, dt='list')
        for cam_name in cam_names[:self.use_camera_num]:
            # Intrinsics
            cam_dict = {}

            cam_dict['K'] = self.read('K_{}'.format(cam_name), True)
            cam_dict['H'] = int(self.read('H_{}'.format(cam_name), True, dt='real')) or -1
            cam_dict['W'] = int(self.read('W_{}'.format(cam_name), True, dt='real')) or -1
            # cam_dict['H'] = 256
            # cam_dict['W'] = 256
            cam_dict['invK'] = np.linalg.inv(cam_dict['K'])

            # Extrinsics
            Tvec = self.read('T_{}'.format(cam_name), False)
            Rvec = self.read('R_{}'.format(cam_name), False)
            if Rvec is not None:
                R = cv2.Rodrigues(Rvec)[0]
            else:
                R = self.read('Rot_{}'.format(cam_name))
                Rvec = cv2.Rodrigues(R)[0]
            RT = np.hstack((R, Tvec))

            cam_dict['R'] = R
            cam_dict["T"] = Tvec
            cam_dict["C"] = - Rvec.T @ Tvec
            cam_dict["RT"] = RT
            cam_dict["Rvec"] = Rvec
            cam_dict["P"] = cam_dict['K'] @ cam_dict['RT']

            if (cam_dict['P'][0, 0] < 0):
                a = 0
            # Distortion
            D = self.read('D_{}'.format(cam_name), True)
            if D is None: D = self.read('dist_{}'.format(cam_name), True)
            cam_dict["D"] = D

            # Time input
            cam_dict['t'] = self.read('t_{}'.format(cam_name), False, dt='real') or 0  # temporal index, might all be 0
            cam_dict['v'] = self.read('v_{}'.format(cam_name), False, dt='real') or 0  # temporal index, might all be 0

            # Bounds, could be overwritten
            cam_dict['n'] = self.read('n_{}'.format(cam_name), False, dt='real') or 0.0001  # temporal index, might all be 0
            cam_dict['f'] = self.read('f_{}'.format(cam_name), False, dt='real') or 1e6  # temporal index, might all be 0
            cam_dict['bounds'] = self.read('bounds_{}'.format(cam_name), False)
            cam_dict['bounds'] = np.array([[-1e6, -1e6, -1e6], [1e6, 1e6, 1e6]]) if cam_dict['bounds'] is None else cam_dict['bounds']

            cam_dict['fov'] = 2 * np.arctan(cam_dict['W'] / (2 * cam_dict['K'][0, 0]))

            # CCM
            cam_dict['ccm'] = self.read('ccm_{}'.format(cam_name), True)
            cam_dict['ccm'] = np.eye(3) if cam_dict['ccm'] is None else cam_dict['ccm']
            mm = -cam_dict['R'].T @ cam_dict['T']
            c2w = np.concatenate([cam_dict['R'].T, mm], axis=-1)
            cam_dict['c2w'] = c2w
            cam_dict['center'] = cam_dict['c2w'][:, -1].reshape(3, 1)
            caminfo = CamParams(**cam_dict)
            self.cameras_all.append(caminfo)

        # handle masks
        self.all_masks = []
        masks_path = os.path.join(cfg.train_dataset['data_root'], 'masks')
        cam_ids = os.listdir(masks_path)
        cam_ids.sort()
        for cam_id in cam_ids[:self.use_camera_num]:
            mask_files = os.listdir(os.path.join(masks_path, cam_id))
            mask_files.sort()
            cam_i_mask = []
            for t_frame, mask_file in enumerate(mask_files):
                mask = imageio.imread(os.path.join(masks_path, cam_id, mask_file))
                mask = np.array(mask).astype(np.float32)
                # mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_AREA) # TODO: remove hardcoding
                cam_i_mask.append(mask)
                if t_frame > self.use_frames - 2:
                    break
            cam_i_mask = np.stack(cam_i_mask)
            self.all_masks.append(cam_i_mask)

        self.all_masks = np.stack(self.all_masks)
        self.mask_max_len_x = 0
        self.mask_max_len_y = 0

        # handle point clouds
        self.all_timestep_pcds = []
        #
        # for i in range(self.all_masks.shape[1]):
        #     mask = torch.tensor(self.all_masks)[:, i, :, :]
        #     # bouding_box=self.extrix_file.getNode("bounds_00").mat()
        #     # voxel_now_step = process_voxels(cfg, self.cameras_all, mask, pcd_index=i)
        #
        #     # def process_voxels(cfg, camera_all=None, all_masks=None, pcd_index=0, use_octree=False):
        #     assert self.cameras_all is not None, "Bounding box is not set"
        #     pcd_path = os.path.join(cfg.train_dataset['data_root'], "surfs")
        #     if not os.path.exists(pcd_path):
        #         os.makedirs(pcd_path)
        #     pcd_path = os.path.join(pcd_path, f"{i:06d}.ply")
        #     if not os.path.exists(pcd_path):
        #         bounding_box = self.cameras_all[0].bounds
        #         iteration = cfg.train_dataset["iteration"]
        #         # cameras=cfg.cameras
        #         voxel_num = cfg.train_dataset['voxel_num_start']
        #         x_lims, y_lims, z_lims = bounding_box[:, 0], bounding_box[:, 1], bounding_box[:, 2]
        #         voxel_now_step, voxel_size = init_voxels(x_lims, y_lims, z_lims, voxel_num)
        #
        #         for i in range(iteration):
        #             for index, cam in enumerate(self.cameras_all):
        #                 mask_now = mask[index, :, :]
        #                 voxel_now_step, voxel_size = set_voxels(voxel_now_step, voxel_size, cam, masks=mask_now)
        #         save_ply(voxel_now_step, pcd_path)
        #     else:
        #         voxel_now_step = read_ply(pcd_path)
        #
        #     self.all_timestep_pcds.append(voxel_now_step)

    def read(self, node, use_intrix=True, dt="mat"):
        if use_intrix:
            fs = self.intrix_file
        else:
            fs = self.extrix_file
        if dt == 'mat':
            output = fs.getNode(node).mat()
        elif dt == 'list':
            results = []
            n = fs.getNode(node)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            output = results
        elif dt == 'real':
            output = fs.getNode(node).real()
        else:
            raise NotImplementedError
        return output

    def get_pcds(self, time_step: int):
        return self.all_timestep_pcds[time_step]

    def get_camera(self, camera_index: int):
        return self.cameras_all[camera_index]

    @property
    def get_camera_length(self):
        return len(self.cameras_all)

    @property
    def get_timestep_length(self):
        return self.use_frames
        # return len(self.all_timestep_pcds)
