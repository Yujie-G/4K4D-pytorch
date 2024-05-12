import torch
import torch.nn as nn
import numpy as np

from lib.config import cfg
from lib.utils.data_utils import normalize
from lib.networks.encoding import get_encoder
from lib.networks.r4K4D.IBR_regressor import ImageBasedSphericalHarmonics as IBR_SH
from lib.networks.r4K4D.Geo_linear import GeoLinear


class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        model_cfg = cfg.model_cfg
        self.pcds = nn.ParameterList() # list of pcd
        self.dim_time = model_cfg.dim_time
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
            # TODO: use InstantNGP to get the init pcd
            pass

    def render_pcd(self, pcd, rgb, rad, density, batch):
        # TODO: implement the differentiable depth peeling
        return None, None, None

    def sample_pcd_pcd_t(self, batch):
        index, time = batch['meta']['latent_index'], batch['t']
        pcd = torch.stack([self.pcds[l] for l in index])  # B, N, 3 # avoid explicit syncing
        pcd_t = time[..., None, None].expand(-1, *pcd.shape[1:-1], 1)  # B, N, 1
        return pcd, pcd_t

    def forward(self, batch):
        pcd, pcd_t = self.sample_pcd_pcd_t(batch)  # B, P, 3, B, P, 1

        xyzt_feat = self.xyzt_encoder(pcd, pcd_t)  # same time

        rad, density = self.geo_linear(xyzt_feat)  # B, N, 1

        self.ibr_encoder(pcd, batch)  # update batch.output
        direction = normalize(pcd.detach() - (-batch.R.mT @ batch.T).mT)  # B, N, 3
        rgb = self.ibr_regressor(torch.cat([xyzt_feat, direction], dim=-1), batch)  # B,  N, 3

        # Perform points rendering
        rgb, acc, dpt = self.render_pcd(pcd, rgb, rad, density, batch)  # B, HW, C

        return pcd, pcd_t, rgb, rad, density
