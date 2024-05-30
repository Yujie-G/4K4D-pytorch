import os
import numpy as np
import imageio
import torch
import torch.utils.data as data

from lib.config import cfg
from lib.utils.camera.optimizable_cam import Camera
from lib.utils.img_utils import save_numpy_image

class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.data_root, self.split = kwargs['data_root'], kwargs['split']
        self.nearset_N_views = cfg.train_dataset['nearset_N_views']
        camera_matrix_optimized_path = os.path.join(self.data_root, 'optimized')
        camera_intri_path = os.path.join(camera_matrix_optimized_path, "intri.yml")
        camera_extri_path = os.path.join(camera_matrix_optimized_path, "extri.yml")
        self.camera = Camera(camera_intri_path, camera_extri_path)
        self.camera_len = self.camera.get_camera_length
        self.time_step_len = self.camera.get_timestep_length

        self.input_ratio = cfg.train_dataset['input_ratio']


        image_path = os.path.join(self.data_root, 'images_calib')
        angles_all = os.listdir(image_path)
        angles_all.sort()
        img = []
        for angle in angles_all:
            image_files = os.listdir(os.path.join(image_path, angle))
            image_files.sort()
            now_angle_images = []
            for index, image_file in enumerate(image_files):
                image = imageio.imread(os.path.join(image_path, angle, image_file))/255
                now_angle_images.append(image)
                if index > self.camera.use_frames - 2:
                    break
            img.append(np.stack(now_angle_images))
        self.img = np.stack(img[:self.camera_len], dtype=np.float32)

    def __getitem__(self, index):

        cam_index = index % self.camera_len
        time_step_index = index // self.camera_len
        cam = self.camera.get_camera(cam_index)
        t_bounds = np.array([0.0, self.time_step_len - 1], dtype=np.float32).reshape(2, -1)
        bounds = cam.bounds
        bounds = np.concatenate([bounds, t_bounds], axis=1)
        # wbounds=wbounds.reshape(-1)
        origin_rgb = self.img[cam_index, time_step_index]
        mask = self.camera.all_masks[cam_index, time_step_index, :, :] / 255.0

        mask_min_x, mask_max_x, mask_min_y, mask_max_y = 0, mask.shape[0], 0, mask.shape[1]
            # np.where(mask > 0)[0].min(), np.where(mask > 0)[0].max(), np.where(mask > 0)[1].min(), np.where(mask > 0)[1].max()

        # uv_rgb = np.array([mask_min_x, mask_min_y])

        rgb = origin_rgb * mask[..., np.newaxis]
        mask = mask[mask_min_x:mask_max_x, mask_min_y:mask_max_y]
        rgb = rgb[mask_min_x:mask_max_x, mask_min_y:mask_max_y, :]
        H = rgb.shape[0]
        W = rgb.shape[1]

        cam_k = cam.K.copy()
        cam_k[0, 2] = cam_k[0, 2] - mask_min_y
        cam_k[1, 2] = cam_k[1, 2] - mask_min_x
        cam_p = cam_k @ cam.RT
        ret = {'rgb': rgb}
        ret.update({'mask': mask})
        ret.update({"time_step": time_step_index, "cam_index": cam_index, "bounds": bounds})
        ret.update({"R": cam.R, "T": cam.T, "K": cam_k, "P": cam_p, "RT": cam.RT, "near": cam.n, "far": cam.f, "fov": cam.fov})
        ret.update({'meta': {'H': H, 'W': W}})

        # handle N nearest views for IBR part
        N_nearest_cam_index, src_exts, _ = self.get_nearest_pose_cameras(cam_index)
        rgb_N_nearest = self.img[N_nearest_cam_index, time_step_index]

        # ret.update({"N_reference_images_index": N_nearest_cam_index})
        masks = self.camera.all_masks[N_nearest_cam_index]

        max_len_x, max_len_y = 0, 0
        cam_k_all = []
        mask_allindex_lists = []
        rgb_N_nearest_list = []
        for index, mask in enumerate(masks):
            mask = mask.squeeze()
            mask_real_index = N_nearest_cam_index[index]
            cam_k_temp = self.camera.get_camera(mask_real_index).K.copy()
            # cam_k_temp=cam.K.copy()
            mask_min_x, mask_max_x, mask_min_y, mask_max_y = 0, mask.shape[0], 0, mask.shape[1]
                # np.where(mask > 0.9)[1].min(), np.where(mask > 0.9)[1].max(), np.where(mask > 0.5)[2].min(), np.where(mask > 0.5)[2].max()
            rgb_N_nearest_list.append(rgb_N_nearest[index, mask_min_x:mask_max_x, mask_min_y:mask_max_y, :])

            cam_k_temp[0, 2] -= mask_min_y
            cam_k_temp[1, 2] -= mask_min_x

            mask_allindex_lists.append([mask_min_x, mask_max_x, mask_min_y, mask_max_y])

            cam_k_all.append(cam_k_temp)
            max_len_x = max(max_len_x, mask_max_x - mask_min_x)
            max_len_y = max(max_len_y, mask_max_y - mask_min_y)
        scale_list = []
        for i, mask_for in enumerate(mask_allindex_lists):
            mask_min_x, mask_max_x, mask_min_y, mask_max_y = mask_for
            scale_list.append(np.array([max_len_x / H, max_len_y / W]))
        scale_list = np.stack(scale_list)
        src_ixts = np.stack(cam_k_all)

        src_inps = []
        for rgb in rgb_N_nearest_list:
            pad_dims = [(0, max_size - img_size) for img_size, max_size in zip(rgb.shape, (max_len_x, max_len_y, 3))]
            rgb_pad = np.pad(rgb, pad_dims, mode='constant')
            src_inps.append(rgb_pad)
        src_inps = np.stack(src_inps)
        ret['meta'].update({"src_inps": src_inps})
        ret.update({"src_exts": src_exts, "src_ixts": src_ixts})
        ret.update({"scale": scale_list})
        return ret

    def get_nearest_pose_cameras(self, now_index):
        target_cam = self.camera.get_camera(now_index)
        all_cams = self.camera.cameras_all
        target_cam_center = target_cam.center
        all_cam_centers = [cam.center for cam in all_cams]
        all_RTs = [cam.RT for cam in all_cams]
        all_Ks = [cam.K for cam in all_cams]

        all_cam_centers = np.stack(all_cam_centers)
        all_RTs = np.stack(all_RTs)
        all_Ks = np.stack(all_Ks)

        all_cam_centers = all_cam_centers.reshape(-1, 3)
        target_cam_center = target_cam_center.reshape(-1, 3)
        all_cam_distance = all_cam_centers - target_cam_center
        all_cam_distance = np.linalg.norm(all_cam_distance, axis=-1)
        distance_index = np.argsort(all_cam_distance, axis=-1)
        distance_index = distance_index[1:self.nearset_N_views + 1]

        RTs_choose = all_RTs[distance_index, :, :].copy()
        KS_choose = all_Ks[distance_index, :, :].copy()

        return distance_index, RTs_choose, KS_choose

    def __len__(self):
        return self.camera_len * self.time_step_len

