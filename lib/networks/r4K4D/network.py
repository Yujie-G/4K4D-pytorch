import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm

from lib.config import cfg
from lib.utils.data_utils import normalize
from lib.utils.ply_utils import read_ply
from lib.networks.encoding import get_encoder
from lib.networks.r4K4D.IBR_regressor import ImageBasedSphericalHarmonics as IBR_SH
from lib.networks.r4K4D.Geo_linear import GeoLinear


class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        self.dim_time = cfg.task_arg.dim_time
        self.cam_num = cfg.task_arg.camera_use
        model_cfg = cfg.model_cfg
        self.pcds = nn.ParameterList() # list of pcd
        self.init_pcds(model_cfg)
        self.xyzt_encoder, feature_dim = get_encoder(model_cfg.network_encoder.xyzt_encoder_cfg)
        self.ibr_encoder = get_encoder(model_cfg.network_encoder.ibr_encoder_cfg)
        self.ibr_regressor = IBR_SH()
        self.geo_linear = GeoLinear(feature_dim, model_cfg.geo_linear)

    def init_pcds(self, cfg):
        skip_loading_points = cfg.skip_loading_points
        if skip_loading_points:
            n_points = cfg.n_points
            for i in range(self.dim_time):
                self.pcds.append(nn.Parameter(torch.as_tensor(torch.rand(n_points, 3, device='cuda')), requires_grad=True))
        else:
            pcd_path = cfg.pcds_dir
            for f in tqdm(range(self.dim_time), desc='Loading pcds to network'):
                pcd = read_ply(os.path.join(pcd_path, f'{f:06d}.ply'))
                self.pcds.append(nn.Parameter(torch.as_tensor(pcd, device='cuda', dtype=torch.float32), requires_grad=True))



    def render_pcd(self, pcd, rgb, rad, density, batch):
        # TODO: implement the differentiable depth peeling
        return None, None, None

    def forward(self, batch):
        # index, time = batch['meta']['latent_index'], batch['time_step']
        # pcd = torch.stack([self.pcds[l] for l in index])  # B, N, 3
        time = batch['time_step']
        pcd = self.pcds[time].unsqueeze(0)   # B, N, 3
        pcd_t = time.view(1, 1).expand(1, pcd.shape[1], 1)  # B, N, 1

        xyzt_feat = self.xyzt_encoder(pcd, pcd_t, batch)  # same time

        rad, density = self.geo_linear(xyzt_feat)  # B, N, 1

        self.ibr_encoder(pcd, batch)  # update batch.output
        direction = normalize(pcd.detach() - (-batch['R'].mT @ batch['T']).mT)  # B, N, 3
        rgb = self.ibr_regressor(torch.cat([xyzt_feat, direction], dim=-1), batch)  # B,  N, 3

        # Perform points rendering
        rgb, acc, dpt = self.render_pcd(pcd, rgb, rad, density, batch)  # B, HW, C

        return pcd, pcd_t, rgb, rad, density
