import torch.utils.data as data
import numpy as np
import os
from lib.utils import data_utils
from lib.config import cfg
from torchvision import transforms as T
import imageio
import json
import cv2


class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene
        self.input_ratio = kwargs['input_ratio']
        self.data_root = os.path.join(data_root, scene)
        self.split = split
        self.batch_size = cfg.task_arg.N_rays

        # read image
        image_paths = []
        poses = []
        json_info = json.load(open(os.path.join(self.data_root, 'transforms_{}.json'.format('train'))))
        for frame in json_info['frames']:
            image_paths.append(os.path.join(self.data_root, frame['file_path'][2:] + '.png'))
            poses.append(np.array(frame['transform_matrix']))

        self.poses = np.array(poses).astype(np.float32)
        self._len = len(image_paths)
        imgs = []
        for image_path in image_paths:
            img = imageio.imread(image_path)/255.
            img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
            if self.input_ratio != 1.:
                img = cv2.resize(img, None, fx=self.input_ratio, fy=self.input_ratio, interpolation=cv2.INTER_AREA)
            imgs.append(img)
        # set images
        self.imgs = np.array(imgs).astype(np.float32) # [N, H, W, 3] such as [100, 400, 400, 3]

        self.H, self.W = self.imgs[0].shape[:2]

        camera_angle_x = float(json_info['camera_angle_x'])
        self.focal = .5 * self.W / np.tan(.5 * camera_angle_x)

        self.K = np.array([
            [self.focal, 0, 0.5 * self.W],
            [0, self.focal, 0.5 * self.H],
            [0, 0, 1]
        ])
        # rays: [N, 2, H, W,  3] --> [N, H, W, 2, 3] (ray origins(xyz) and directions(xyz))
        self.rays = np.stack([np.stack(self.get_rays(self.H, self.W, self.K, p)) for p in self.poses[:, :3, :4]], 0)
        self.rays = np.transpose(self.rays, [0, 2, 3, 1, 4])

    def __getitem__(self, index):
        if self.split == 'train':
            ids = np.random.choice(self._len, self.batch_size, replace=False)
            gt = self.imgs.reshape(-1, 3)[ids]
            rays = self.rays.reshape(-1, 2, 3)[ids]
        else:
            gt = self.imgs.reshape(-1, 3)
            rays = self.rays.reshape(-1, 2, 3)
        ret = {'gt': gt, 'rays': rays} # input and output. they will be sent to cuda
        ret.update({'meta': {'H': self.H, 'W': self.W}}) # meta means no need to send to cuda
        return ret

    def __len__(self):
        return self._len


    def get_rays(self, H, W, K, c2w):
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
        return rays_o, rays_d


