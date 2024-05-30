import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm

from lib.config import cfg
from lib.utils.data_utils import normalize
from lib.utils.ply_utils import read_ply
from lib.utils.img_utils import save_tensor_image
from lib.utils.base_utils import dotdict
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
        self.ibr_encoder, _ = get_encoder(model_cfg.network_encoder.ibr_encoder_cfg)
        self.ibr_regressor = IBR_SH(model_cfg.IBR_regressor)
        self.geo_linear = GeoLinear(feature_dim, model_cfg.geo_linear)
        self.K_points = model_cfg.K_points

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

    def prepare_opengl(self,
                       module_attribute: str = 'cudagl',
                       renderer_class=None,
                       dtype: torch.dtype = torch.half,
                       tex_dtype: torch.dtype = torch.half,
                       H: int = 1024,
                       W: int = 1024,
                       size: int = 262144,
                       ):
        # Lazy initialization of EGL context
        if 'eglctx' not in cfg and 'window' not in cfg:
            # log(f'Init eglctx with h, w: {H}, {W}')
            from lib.utils.egl_utils import eglContextManager, common_opengl_options
            self.eglctx = eglContextManager(W, H)
            common_opengl_options()


        # Lazy initialization of cuda renderer and opengl buffers, this is only a placeholder
        if not hasattr(self, module_attribute):
            rand1 = torch.rand(size, 1, dtype=dtype)
            rand3 = torch.rand(size, 3, dtype=dtype)
            opengl = renderer_class(verts=rand3,  # !: BATCH
                                    colors=rand3,
                                    scalars=dotdict(radius=rand1, alpha=rand1),
                                    pts_per_pix=self.K_points,
                                    render_type=renderer_class.RenderType.POINTS,
                                    dtype=dtype,
                                    tex_dtype=tex_dtype,
                                    H=H,
                                    W=W)  # this will preallocate sizes
            setattr(self, module_attribute, opengl)

    def render_pcd(self, pcd, rgb_feat, rad, density, batch, use_opengl=False):
        """
            input: pcd, rgb_feat, rad, density, batch
            return: rgb, acc, dpt
        """
        rgb, acc, dpt = None, None, None
        if use_opengl:
            from lib.networks.r4K4D.renderer import openglRenderer
            self.prepare_opengl('cudagl', openglRenderer, torch.float32, torch.float32,
                                batch['meta']['H'],
                                batch['meta']['W'], pcd.shape[1])
            self.cudagl: openglRenderer
            self.cudagl.pts_per_pix = self.K_points
            self.cudagl.volume_rendering = True
            rgb, acc, dpt = self.cudagl.forward(pcd, rgb_feat, rad, density, batch)
        else:
            from lib.networks.r4K4D.renderer import torchRender
            rgb, acc, dpt = torchRender(pcd, rgb_feat, rad, density, H=batch['meta']['H'], W=batch['meta']['W'], K=batch['K'], R=batch['R'], T=batch['T'], K_points=self.K_points)
        return rgb, acc, dpt

    def forward(self, batch):
        # index, time = batch['meta']['latent_index'], batch['time_step']
        # pcd = torch.stack([self.pcds[l] for l in index])  # B, N, 3
        time = batch['time_step']
        pcd = self.pcds[time].unsqueeze(0)   # B, N, 3
        pcd_t = time.view(1, 1).expand(1, pcd.shape[1], 1)  # B, N, 1

        xyzt_feat = self.xyzt_encoder(pcd, pcd_t, batch)  # same time

        rad, density = self.geo_linear(xyzt_feat)  # B, N, 1

        self.ibr_encoder(pcd, batch)  # update batch['output']
        direction = normalize(pcd.detach() - (-batch['R'].transpose(-2, -1) @ batch['T']).transpose(-2, -1))  # B, N, 3
        rgb = self.ibr_regressor(torch.cat([xyzt_feat, direction], dim=-1), batch)  # B, N, 3

        # # DEBUG:
        # rgb = torch.ones(rgb.shape, device=rgb.device)  # B, N, 3
        # rad = 0.005 * torch.ones(rad.shape, device=rad.device)
        # density = torch.ones(density.shape, device=density.device)


        # Perform points rendering
        rgb_map, acc, dpt = self.render_pcd(pcd, rgb, rad, density, batch)  # B, HW, C

        ret = {'rgb': rgb_map, 'acc': acc, 'dpt': dpt, 'mask': batch['mask']}
        return ret
