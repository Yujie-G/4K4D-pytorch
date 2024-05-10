import torch
import torch.nn as nn
from torch.nn import functional as F
from lib.config import cfg, args
from lib.utils.net_utils import make_buffer

# TODO: DEBUG THIS
class HexPlane(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        n_features_per_level = kwargs['n_features_per_level']
        n_levels = kwargs['n_levels']
        b = kwargs['b']
        base_resolution = kwargs['base_resolution']
        dim_time = kwargs['dim_time']
        bounds = kwargs['bounds']
        std = 1e-1
        self.bounds = make_buffer(torch.as_tensor(bounds, dtype=torch.float))

        # use torch as backbone
        self.spatial_embedding = nn.ParameterList([
            nn.Parameter(
                torch.zeros(3, n_features_per_level, int(base_resolution * b ** i), int(base_resolution * b ** i)))
            # [xy, xz, yz], C, H, H
            for i in range(n_levels)])
        self.temporal_embedding = nn.ParameterList([
            nn.Parameter(torch.zeros(3, n_features_per_level, int(base_resolution * b ** i), dim_time))
            # [xy, xz, yz], C, H, T
            for i in range(n_levels)])
        for data in self.spatial_embedding:
            data.data.uniform_(-std, std)
        for data in self.temporal_embedding:
            data.data.uniform_(-std, std)


        self.xy = make_buffer(torch.as_tensor([0, 1], dtype=torch.long))  # to avoid synchronization
        self.xz = make_buffer(torch.as_tensor([0, 2], dtype=torch.long))  # to avoid synchronization
        self.yz = make_buffer(torch.as_tensor([1, 2], dtype=torch.long))  # to avoid synchronization

    def forward(self, xyz: torch.Tensor, t: torch.Tensor, batch):
        bash = xyz.shape  # batch shape
        xyz = xyz.view(-1, xyz.shape[-1])
        t = t.reshape(-1, t.shape[-1])
        xyz = (xyz - self.bounds[0]) / (self.bounds[1] - self.bounds[0])  # normalized, N, 3

        # get, xy, xz, yz, tx, ty, tz
        spatial_coords = torch.stack([xyz[..., self.xy.long()], xyz[..., self.xz.long()], xyz[..., self.yz.long()]],
                                     dim=0) * 2. - 1.  # [xy, xz, yz] from [0, 1] -> [-1, 1] -> 3, N, 2
        temporal_coords = torch.stack([torch.cat([t, xyz[..., :1]], dim=-1),  # tx
                                       torch.cat([t, xyz[..., 1:2]], dim=-1),  # ty
                                       torch.cat([t, xyz[..., 2:3]], dim=-1)],
                                      dim=0) * 2. - 1.  # tz -> [tx, ty, tz] from [0, 1] -> [-1, 1] -> 3, N, 2

        spatial_feats = []
        temporal_feats = []
        # NOTE: About the arrangement of temporal index
        # NOTE: The dataset provides a t that maps 0 -> 1 to 0 -> 99 (0 correspondes to the first frame, 99 to the last frame)
        # NOTE: Thus this mimics a sampling of align corners == True
        for data in self.spatial_embedding:
            # 3, 1, N, 2 -> 3, C, 1, N -> 3, C, N -> 3, N, C
            feat = F.grid_sample(data, spatial_coords[:, None], mode='bilinear', padding_mode='border',
                                 align_corners=False)[:, :, 0].permute((0, 2, 1))  # xy need to be reverted
            spatial_feats.append(feat)
        for data in self.temporal_embedding:
            # 3, 1, N, 2 -> 3, C, 1, N -> 3, C, N -> 3, N, C
            feat = F.grid_sample(data, temporal_coords[:, None], mode='bilinear', padding_mode='border',
                                 align_corners=True)[:, :, 0].permute((0, 2, 1))  # xy need to be reverted
            temporal_feats.append(feat)
        spatial_feat = torch.cat(spatial_feats, dim=-1)  # 3, N, 4C
        temporal_feat = torch.cat(temporal_feats, dim=-1)  # 3, N, 4C
        feat = torch.cat([spatial_feat, temporal_feat], dim=0)  # 6, N, 4C

        val = torch.cat([item for item in feat], dim=-1)  # N, 24C

        return val.view(*bash[:-1], val.shape[-1])  # B, N, 24C (or 4c)



