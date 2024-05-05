import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.networks.encoding import get_encoder
from lib.config import cfg

class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        net_cfg = cfg.network
        self.xyz_encoder, xyz_input_ch = get_encoder(net_cfg.xyz_encoder)
        self.viewdir_encoder, dir_input_ch = get_encoder(net_cfg.dir_encoder)
        D, W  = net_cfg.nerf.D, net_cfg.nerf.W
        self.pos_layers = nn.ModuleList(
            [nn.Linear(xyz_input_ch, W).double()] + [nn.Linear(W, W).double() for _ in range(D-1)])
        self.viewdir_layers = nn.ModuleList(
            [nn.Linear(dir_input_ch + W, W//2).double()])
        self.feature_layers = nn.Linear(W, W).double()
        self.alpha_layers = nn.Linear(W, 1).double()
        self.rgb_layers = nn.Linear(W // 2, 3).double()

    def network_query(self, xyz, viewdirs):
        xyz_flat = xyz.reshape(-1, xyz.shape[-1])
        xyz_encoded = self.xyz_encoder(xyz_flat)
        viewdirs = viewdirs[:, None].expand(xyz.shape[0], xyz.shape[1], viewdirs.shape[-1])
        viewdirs = viewdirs.reshape(-1, viewdirs.shape[-1])
        viewdirs_encoded = self.viewdir_encoder(viewdirs)
        h = xyz_encoded
        for i, l in enumerate(self.pos_layers):
            h = l(h)
            h = F.relu(h)
        alpha = self.alpha_layers(h)
        feature = self.feature_layers(h)

        h = torch.cat([feature, viewdirs_encoded], -1)
        for i, l in enumerate(self.viewdir_layers):
            h = l(h)
            h = F.relu(h)
        rgb = self.rgb_layers(h)
        return torch.cat([rgb, alpha], -1)

    def render(self, ray_batch, batch):
        rays_o = ray_batch[:, 0, :].squeeze() # [N_rays, 3]
        rays_d = ray_batch[:, 1, :].squeeze() # [N_rays, 3]
        N_rays = rays_o.shape[0]
        near, far, cascade_samples = batch['meta']['near'], batch['meta']['far'], batch['meta']['cascade_samples']
        N_importance, N_samples = cascade_samples[0].item(), cascade_samples[1].item()

        sample_pts, z_vals = self.sample(rays_o, rays_d, near, far, N_importance, N_samples)
        view_dirs = rays_d
        raw = self.network_query(sample_pts, view_dirs)
        raw = raw.reshape(N_rays, N_importance, 4)
        output = self.raw2outputs(raw, z_vals, rays_d)
        return output

    def batchify(self, rays, batch):
        all_ret = {}
        chunk = cfg.task_arg.chunk_size
        for i in range(0, rays.shape[0], chunk):
            ret = self.render(rays[i:i + chunk], batch)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], dim=0) for k in all_ret}
        return all_ret

    def sample(self, rays_o, rays_d, near, far, N_samples, N_importance):
        # TODO: sample rays
        N_rays = rays_o.shape[0]
        t_vals = torch.linspace(0., 1., steps=N_samples)
        z_vals = near * (1. - t_vals) + far * (t_vals)
        z_vals = z_vals.expand([N_rays, N_samples])

        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        z_vals = lower + (upper - lower) * t_rand
        z_vals = z_vals.cuda()
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        return pts, z_vals

    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).cuda().expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
                noise = torch.Tensor(noise)

        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), dtype=torch.float32).cuda(), 1. - alpha + 1e-10], -1), -1)[:,
                          :-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return {'rgb': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map, 'weights': weights, 'depth_map': depth_map}


    def forward(self, batch):
        batch['rays'] = batch['rays'].squeeze()
        N_rays, _, C = batch['rays'].shape
        ret = self.batchify(batch['rays'].reshape(-1, 2, C), batch)
        return {k:ret[k].reshape(N_rays, -1) for k in ret}
