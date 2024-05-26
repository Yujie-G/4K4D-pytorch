import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision.io import decode_jpeg

from lib.utils.net_utils import FeatureNet
from lib.utils.img_utils import pad_image


class GeometryImageBasedEmbedder(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.n_track = 8
        self.feat_reg = FeatureNet()
        self.img_pad = self.feat_reg.size_pad
        self.src_dim = self.feat_reg.out_dims[-1] + 3  # image feature and rgb color for blending
        self.out_dim = None

        self.use_interpolate = False
        self.opt_cnn_warmup = 1000

    def compute_src_inps(self, batch: dict, key: str = 'src_inps'):
        if key not in batch and \
                key in batch['meta'] and \
                isinstance(batch['meta'][key], list) and \
                batch['meta'][key][0].ndim == 2:
            batch[key] = [torch.cat([decode_jpeg(inp.cpu(), device='cuda') for inp in inps])[None].float() / 255 \
                          for inps in batch['meta'][key]]
        if isinstance(batch[key], list) and batch[key][0].ndim == 4:
            max_h = max([i.shape[-2] for i in batch[key]])
            max_w = max([i.shape[-1] for i in batch[key]])
            batch[key] = torch.stack([pad_image(img, [max_h, max_w]) for img in batch[key]]).permute(1, 0, 2, 3, 4)
        return batch[key].contiguous()

    def compute_src_feats(self, batch: dict):
        # Prepare inputs and feature
        # src_inps = self.compute_src_inps(batch).to(self.feat_reg.conv0[0].conv.weight.dtype)
        src_inps = batch['meta']['src_inps'][0].permute(0,3,1,2).unsqueeze(0).to(self.feat_reg.conv0[0].conv.weight.dtype).cuda()
        # Values to be reused
        # Preparing source scaling (for painless up convolution and skip connections)
        Hc, Wc = src_inps.shape[-2:]  # cropped image size
        Hp, Wp = int(np.ceil(Hc / self.img_pad)) * self.img_pad, int(
            np.ceil(Wc / self.img_pad)) * self.img_pad  # Input and output should be same in size
        src_inps = pad_image(src_inps, size=[Hp, Wp])  # B, S, 3, H, W

        # Preparing source image scaling
        if self.use_interpolate:
            src_scale = src_inps.new_empty(2, 1)
            src_scale[0] = Wp / Wc
            src_scale[1] = Hp / Hc
        else:
            src_scale = src_inps.new_ones(2, 1)  # 2, 1

        # Forward feature extraction
        def feature_extraction(src_inps: torch.Tensor, feat_reg: nn.Module, batch):
            sh = src_inps.shape
            src_inps = src_inps.view(-1, *sh[-3:])
            feats = feat_reg(src_inps * 2 - 1)  # always return a tensor
            feats = [f.view(*sh[:-3], *f.shape[-3:]) for f in feats]
            return feats

        # `src_feats` is a list of features of shape (B, S, C, H, W) -> (B, S, 32*(2**(-i)), H//4*(2**i), W//4*(2**i))
        feats = []
        for i, inp in enumerate(src_inps[0]):  # !: BATCH
            if i < self.n_track:
                feats.append(feature_extraction(inp, self.feat_reg, batch)[-1])  # C, H, W
            else:
                with torch.no_grad():  # no gradient tracking for these images, to save training time
                    feats.append(feature_extraction(inp, self.feat_reg, batch)[-1])
        src_feat = torch.stack(feats)[None]  # S, C, H, W -> B, S, C, H, W
        src_feat_rgb = torch.cat([src_feat, src_inps], dim=-3)  # B, S, C, Hr, Wr
        if 'persistent' not in batch:
            batch['persistent'] = {}
        batch['persistent']['src_feat_rgb'] = src_feat_rgb
        batch['persistent']['src_scale'] = src_scale

    def sample_geometry_feature_image(self, xyz: torch.Tensor,  # B, P, 3
                                      src_feat_rgb: torch.Tensor,  # B, S, C, H, W
                                      src_exts: torch.Tensor,  # B, S, 3, 4
                                      src_ixts: torch.Tensor,  # B, S, 3, 3
                                      src_scale: torch.Tensor,
                                      padding_mode: str = 'border',
                                      ):
        # xyz: B, P, 3
        # src_feat_rgb: B, S, C, Hs, Ws
        B, S, C, Hs, Ws = src_feat_rgb.shape
        B, P, _ = xyz.shape
        xyz1 = torch.cat([xyz, torch.ones_like(xyz[..., -1:])], dim=-1)  # homogeneous coordinates

        src_ixts = src_ixts.clone()
        src_ixts[..., :2, :] *= src_scale

        # B, P, 4 -> B, 1, P, 4
        # B, S, 4, 4 -> B, S, 4, 4
        # -> B, S, P, 4
        xyz1 = (xyz1[..., None, :, :] @ src_exts.transpose(-2, -1))
        xyzs = xyz1[..., :3] @ src_ixts.transpose(-2, -1)  # B, S, P, 3 @ B, S, 3, 3
        xy = xyzs[..., :-1] / (xyzs[..., -1:] + 1e-8)  # B, S, P, 2
        x, y = xy.chunk(2, dim=-1)  # B, S, P, 1
        xy = torch.cat([x / Ws * 2 - 1, y / Hs * 2 - 1], dim=-1)  # B, S, P, 2

        # Actual sampling of the image features (along with rgb colors)
        # BS, C, 1, P -> B, S, C, P -> B, S, P, C
        src_feat_rgb = F.grid_sample(src_feat_rgb.view(-1, C, Hs, Ws), xy.view(-1, 1, P, 2),
                        padding_mode=padding_mode).view(B, S, C, P).permute(0, 1, 3, 2)
        return src_feat_rgb

    def forward(self, xyz: torch.Tensor, batch: dict):
        # xyz: B, P * S, 3
        if 'iter' not in batch['meta']:
            batch['meta']['iter'] = 0
        else:
            batch['meta']['iter'] += 1
        if batch['meta']['iter'] > self.opt_cnn_warmup:
            with torch.no_grad():
                self.compute_src_feats(batch)
        else:
            self.compute_src_feats(batch)
        # Extract things from batch
        src_feat_rgb = batch['persistent']['src_feat_rgb']  # last level of source image feature, B, S, C, Hs, Ws
        src_scale = batch['persistent']['src_scale']

        # the source camera parameter is still required to perform the sampling of nearby points...
        src_exts, src_ixts = batch['src_exts'], batch['src_ixts']

        # Sample image feature
        src_feat_rgb = self.sample_geometry_feature_image(xyz, src_feat_rgb, src_exts, src_ixts, src_scale)  # B, S, P, C

        # Store output to batch variable, this module is not intended to be inserted as a normal regressor
        batch['output'] = {}
        batch['output']['src_feat_rgb'] = src_feat_rgb  # B, S, P, C
        return src_feat_rgb  # not to intendied to be used directly # MARK: No dir needed
