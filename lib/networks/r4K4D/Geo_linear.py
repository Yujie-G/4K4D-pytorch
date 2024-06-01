import torch
import torch.nn as nn

class GeoLinear(nn.Module):
    def __init__(self, feature_dim, cfg):
        super(GeoLinear, self).__init__()
        W, D = cfg.width, cfg.depth

        linear = []
        linear.append(nn.Linear(feature_dim, W))
        for i in range(D - 1):
            linear.append(nn.Linear(W, W))
        linear.append(nn.Linear(W, 2))
        self.linear = nn.ModuleList(linear)
        self.out_dim = 2

    def geo_actvn(self, x: torch.Tensor,
                  radius_min: float,
                  radius_max: float,
                  radius_shift: float,
                  density_shift: float):
        r, dens = x.split([1, 1], dim=-1)
        r = (r + radius_shift).sigmoid() * (radius_max - radius_min) + radius_min
        dens = (dens + density_shift).sigmoid()
        return r, dens

    def forward(self, feat):
        h = feat
        for layer in self.linear:
            h = layer(h)

        r, dens = self.geo_actvn(h, 0.001, 0.015, 0.0, 0.0)
        r_max = r.max()
        r_min = r.min()
        dens_max = dens.max()
        dens_min = dens.min()
        return r, dens