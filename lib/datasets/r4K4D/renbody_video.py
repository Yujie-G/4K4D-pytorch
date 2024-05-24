import os
import numpy as np
import imageio
import torch.utils.data as data

from lib.config import cfg
from lib.utils.camera.optimizable_cam import Camera
from lib.utils.img_utils import pad_image


class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.data_root, self.split = kwargs['data_root'], kwargs['split']
        self.nearset_num = cfg.train_dataset['nearset_num']
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
                image = imageio.imread(os.path.join(image_path, angle, image_file)) / 255
                # image=resize_array(image,(256,256,3))
                now_angle_images.append(image)
                if index > self.camera.use_frames - 2:
                    break
            img.append(np.stack(now_angle_images))
        self.img = np.stack(img[:self.camera_len])
        # # set uv
        # H, W = self.img.shape[-3:-1]
        # self.H, self.W = H, W
        # X, Y = np.meshgrid(np.arange(W), np.arange(H))
        # u, v = X.astype(np.float32) / (W - 1), Y.astype(np.float32) / (H - 1)
        # self.uv = np.stack([u, v], -1).reshape(-1, 2).astype(np.float32)

    def __getitem__(self, index):
        # index=21

        cam_index = index % self.camera_len
        time_step_index = index // self.camera_len
        cam = self.camera.get_camera(cam_index)
        # pcd=self.camera.get_pcds(time_step_index)
        # pcd=pcd.reshape(-1,3)
        t_bounds = np.array([0.0, self.time_step_len - 1]).reshape(2, -1)
        wbounds = cam.bounds
        wbounds = np.concatenate([wbounds, t_bounds], axis=1)
        # wbounds=wbounds.reshape(-1)
        rgb = self.img[cam_index, time_step_index]
        mask = self.camera.all_masks[cam_index, time_step_index, :, :] / 255

        mask_min_x, mask_max_x, mask_min_y, mask_max_y = np.where(mask > 0)[0].min(), np.where(mask > 0)[0].max(), \
        np.where(mask > 0)[1].min(), np.where(mask > 0)[1].max()

        uv_rgb = np.array([mask_min_x, mask_min_y])

        rgb = rgb * mask[..., np.newaxis]
        mask = mask[mask_min_x:mask_max_x, mask_min_y:mask_max_y]
        rgb = rgb[mask_min_x:mask_max_x, mask_min_y:mask_max_y, :]
        H = rgb.shape[0]
        W = rgb.shape[1]

        cam_k = cam.K.copy()
        # print(cam_k)
        # cam_k[0,...]=cam_k[0,...]*(mask_max_y-mask_min_y)/self.W
        # cam_k[1,...]=cam_k[1,...]*(mask_max_x-mask_min_x)/self.H
        cam_k[0, 2] = cam_k[0, 2] - mask_min_y
        cam_k[1, 2] = cam_k[1, 2] - mask_min_x
        cam_p = cam_k @ cam.RT
        ret = {'rgb': rgb}  # input and output. they will be sent to cuda
        ret.update({'mask': mask})
        ret.update({'cam': cam, "time_step": time_step_index, "cam_index": cam_index,
                    "wbounds": wbounds})
        # ret.update({'rays_o': cam.T})
        ret.update({"R": cam.R, "T": cam.T, "K": cam_k, "P": cam_p, "RT": cam.RT, "near": cam.n, "far": cam.f, "fov": cam.fov})
        ret.update({'meta': {'H': H, 'W': W}})
        N_reference_images_index, RTS, KS = self.get_nearest_pose_cameras(cam_index)
        rgb_reference_images = self.img[N_reference_images_index, time_step_index]

        ret.update({"N_reference_images_index": N_reference_images_index})
        masks = self.camera.all_masks[N_reference_images_index]
        rgb_reference_images_list = []

        smallest_uv = []
        max_len_x, max_len_y = 0, 0
        cam_k_all = []
        mask_allindex_lists = []
        for index, mask in enumerate(masks):
            mask_real_index = N_reference_images_index[index]
            cam_k_temp = self.camera.get_camera(mask_real_index).K.copy()
            # cam_k_temp=cam.K.copy()
            mask_min_x, mask_max_x, mask_min_y, mask_max_y = np.where(mask > 0.9)[1].min(), np.where(mask > 0.9)[
                1].max(), np.where(mask > 0.5)[2].min(), np.where(mask > 0.5)[2].max()
            rgb_reference_images_list.append(
                rgb_reference_images[index, mask_min_x:mask_max_x, mask_min_y:mask_max_y, :])
            uv_temp = np.array([mask_min_x, mask_min_y])
            smallest_uv.append(uv_temp)
            cam_k_temp[0, 2] -= mask_min_y
            cam_k_temp[1, 2] -= mask_min_x

            mask_allindex_lists.append([mask_min_x, mask_max_x, mask_min_y, mask_max_y])

            cam_k_all.append(cam_k_temp)
            if (mask_max_x - mask_min_x > max_len_x):
                max_len_x = mask_max_x - mask_min_x
            if (mask_max_y - mask_min_y > max_len_y):
                max_len_y = mask_max_y - mask_min_y
        # max_len_x=math.ceil(max_len_x/8)*8
        # max_len_y=math.ceil(max_len_y/8)*8
        scale_list = []
        for i, mask_for in enumerate(mask_allindex_lists):
            mask_min_x, mask_max_x, mask_min_y, mask_max_y = mask_for
            scale1 = max_len_x / H
            scale2 = max_len_y / W
            scale = np.array([scale1, scale2])
            scale_list.append(scale)
            # len_pad_x=max(max_len_x-(mask_max_x-mask_min_x),0)
            # len_pad_y=max(max_len_y-(mask_max_y-mask_min_y),0)
            # cam_k_all[i][0,2]-=80
            # cam_k_all[i][1,2]-=80
        scale_list = np.stack(scale_list)
        K_for_reference = np.stack(cam_k_all)
        # TODO 这里可能是错误的
        # K_for_reference=cam.K.copy()

        # K_for_reference[:,0,:]=K_for_reference[:,0,:]*max_len_x/self.H
        # K_for_reference[:,1,:]=K_for_reference[:,1,:]*max_len_y/self.W
        rgb_reference_images_list_final = []
        for rgb_reference_image in rgb_reference_images_list:
            pad_dims = [(0, max_size - img_size) for img_size, max_size in zip(rgb_reference_image.shape, (max_len_x, max_len_y, 3))]
            rgb_reference_image_temp = np.pad(rgb_reference_image, pad_dims, mode='constant')
            rgb_reference_images_list_final.append(rgb_reference_image_temp)
        rgb_reference_images_list_final = np.stack(rgb_reference_images_list_final)
        # K_for_reference=K_for_reference[np.newaxis,...]
        projections = K_for_reference @ RTS
        ret.update({"rgb_reference_images": rgb_reference_images_list_final})
        ret.update({"projections": projections})
        ret.update({"uv_rgb": uv_rgb})
        ret.update({"refernce_k": K_for_reference, "refernce_RTs": RTS})
        ret.update({"scale": scale_list})
        # ret.update({"smallest_uv":smallest_uv})

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
        distance_index = distance_index[1:self.nearset_num + 1]

        # distance_index[:]=21

        RTs_choose = all_RTs[distance_index, :, :].copy()
        KS_choose = all_Ks[distance_index, :, :].copy()

        return distance_index, RTs_choose, KS_choose

    def __len__(self):
        return self.camera_len * self.time_step_len

