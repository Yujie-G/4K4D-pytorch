import torch
import torch.nn as nn

from lib.utils.net_utils import MLP as MlpRegressor
from lib.utils.img_utils import eval_sh

class ImageBasedSphericalHarmonics(nn.Module):
    def __init__(self,cfg) -> None:
                 # sh_deg: int = 3,
                 # in_dim: int = 192,  # feature channel dim (vox + img?)
                 # src_dim: int = 8 + 3,
                 # out_dim: int = 3,
                 # width: int = 64,
                 # depth: int = 1,  # use small regressor network
                 # resd_limit: float = 0.25,
                 # resd_init: float = 0.0,
                 # resd_weight_init: float = 0.01,
                 # manual_chunking: bool = False,
                 # **kwargs,

        super().__init__()
        self.sh_deg = cfg.sh_deg
        self.sh_dim = (self.sh_deg + 1) ** 2 * cfg.out_dim
        self.in_dim = cfg.in_dim # feature channel 32 * 6
        self.out_dim = cfg.out_dim
        self.src_dim = cfg.src_dim # ibr_rgb_feat + 3

        self.resd_limit = cfg.resd_limit if hasattr(cfg, 'resd_limit') else 0.25
        self.rgb_mlp = MlpRegressor(self.in_dim + self.src_dim, 1, cfg.width, cfg.depth, out_actvn=nn.Identity(), actvn='relu')  # blend weights
        self.sh_mlp = MlpRegressor(self.in_dim, self.sh_dim, cfg.width, cfg.depth, out_actvn=nn.Identity(), actvn='softplus')
        [self.sh_mlp.layers[i].weight.data.normal_(0, 0.01) for i in range(len(self.sh_mlp.layers))]
        [self.sh_mlp.layers[i].bias.data.fill_(0.0 if i == len(self.sh_mlp.layers) - 1 else -1) for i in range(len(self.sh_mlp.layers))]

    def forward(self, xyzt_feat_dir: torch.Tensor, batch):
        # geo_feat: B, P, C # vox(8) + img(16) + geo(64)?
        xyzt_feat, dir = xyzt_feat_dir[..., :-3], xyzt_feat_dir[..., -3:]  # extract view direction

        # Prepare for directional feature
        ibr_feat: torch.Tensor = batch['output']['src_feat_rgb']  # B, S, P, C
        ibr_rgbs = ibr_feat[..., -3:]  # -4: dir feat, -7 -> -3: rgb, B, S, P, 3

        # Prepare for image based rendering blending (in a narrow sense)
        B, S, P, _ = ibr_feat.shape

        exp_xyz_feat = xyzt_feat[:, None].expand(ibr_feat.shape[:-1] + (xyzt_feat.shape[-1],))  # B, S, P, C
        feat = torch.cat([exp_xyz_feat, ibr_feat], dim=-1)  # +7, append the actual image feature
        rgb_bws = self.rgb_mlp(feat)

        rgb_bws = rgb_bws.softmax(-3)  # B, S, P, 1
        rgb = (ibr_rgbs * rgb_bws).sum(-3)  # B, P, 3, now we have the base rgb

        sh = self.sh_mlp(xyzt_feat)  # only available for geometry rendering
        sh = sh.view(*sh.shape[:-1], self.out_dim, self.sh_dim // self.out_dim)  # reshape to B, P, 3, SH

        # Evaluation of specular SH and base image based rgb
        rgb = rgb + eval_sh(self.sh_deg, sh, dir).tanh() * self.resd_limit  # B, P, 3
        return rgb.clip(0, 1)