import torch
import torch.nn as nn

class GeoLinear(nn.Module):
    def __init__(self, feature_dim, cfg):
        super(GeoLinear, self).__init__()
        W, D = cfg.width, cfg.depth
        out_dim = cfg.out_dim
        self.linear = nn.ModuleList(
            [nn.Linear(feature_dim, W) for _ in range(D)] + [nn.Linear(W, out_dim)])

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

        r, a = self.geo_actvn(h, 0.001, 0.015, -5.0, 5.0)
        return r, a