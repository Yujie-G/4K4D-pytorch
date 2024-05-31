import torch
import cv2
import os
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

from lib.utils.camera.r4k4d_cam_param import CamParams
from lib.utils.ply_utils import *

class Camera:
    def __init__(self, cam_path_intri, cam_path_extri, camera_num, frames_num, data_root):
        self.data_root = data_root
        self.intrix_file = cv2.FileStorage(cam_path_intri, cv2.FILE_STORAGE_READ)
        self.extrix_file = cv2.FileStorage(cam_path_extri, cv2.FILE_STORAGE_READ)
        self.camera_len = camera_num
        self.time_step_len = frames_num

        self.cameras_all = []
        cam_names = self.read('names', True, dt='list')
        for cam_name in cam_names[:self.camera_len]:
            # Intrinsics
            cam_dict = {}

            cam_dict['K'] = self.read('K_{}'.format(cam_name), True)
            cam_dict['H'] = int(self.read('H_{}'.format(cam_name), True, dt='real')) or -1
            cam_dict['W'] = int(self.read('W_{}'.format(cam_name), True, dt='real')) or -1
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
        masks_path = os.path.join(self.data_root, 'masks')
        cam_ids = sorted(os.listdir(masks_path))

        print("Loading masks")

        for cam_id in tqdm(cam_ids[:self.camera_len]):
            cam_id_path = os.path.join(masks_path, cam_id)
            mask_files = sorted(os.listdir(cam_id_path))[:self.time_step_len]

            cam_i_mask = [np.array(imageio.imread(os.path.join(cam_id_path, mask_file))).astype(np.float32) for
                          mask_file in mask_files]
            self.all_masks.append(np.stack(cam_i_mask))

        self.all_masks = np.stack(self.all_masks)

        img = []
        image_path = os.path.join(self.data_root, 'images_calib')
        angle_paths = [os.path.join(image_path, angle) for angle in sorted(os.listdir(image_path))[:self.camera_len]]
        print("Loading images")
        for angle_path in tqdm(angle_paths):
            image_files = sorted(os.listdir(angle_path))[:self.time_step_len]
            now_angle_images = list(
                map(lambda image_file: imageio.imread(os.path.join(angle_path, image_file)) / 255.0, image_files))
            img.append(np.stack(now_angle_images))
        self.imgs = np.stack(img, dtype=np.float32)

    def read(self, node, use_intrix=True, dt="mat"):
        if use_intrix:
            fs = self.intrix_file
        else:
            fs = self.extrix_file
        if dt == 'mat':
            output = fs.getNode(node).mat().astype(np.float32)
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

    def get_camera(self, camera_index: int):
        return self.cameras_all[camera_index]
