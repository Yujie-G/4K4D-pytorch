import torch
import gc
import torch.nn as nn
from lib.config import cfg
from lpips import LPIPS
from lib.utils.img_utils import save_tensor_image

class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.color_crit = nn.MSELoss(reduction='mean')
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

        self.perc_loss_weight = cfg.train.lmbda1
        self.msk_loss_weight = cfg.train.lmbda2
        self.lpips = LPIPS(net='vgg')

    def forward(self, batch):
        output = self.net(batch)
        save_tensor_image(output['rgb'].squeeze(0)*255, 'single-output_rgb.png')
        save_tensor_image(batch['rgb'].squeeze(0)*255, 'single-gt_rgb.png')

        scalar_stats = {}
        loss = 0
        color_loss = self.color_crit(output['rgb'], batch['rgb'])
        scalar_stats.update({'color_mse': color_loss})
        loss += color_loss
        psnr = -10. * torch.log(color_loss.detach()) / \
               torch.log(torch.Tensor([10.]).to(color_loss.device))
        scalar_stats.update({'psnr': psnr})

        lpips_loss = self.lpips(output['rgb'].permute(0, 3, 1, 2), batch['rgb'].permute(0, 3, 1, 2)).squeeze()
        scalar_stats.update({'lpips': lpips_loss})
        loss += self.perc_loss_weight * lpips_loss
        output_rgb_msk = output['rgb']*(output['mask'].unsqueeze(-1))


        gt_rgb_msk = batch['rgb']*(batch['mask'].unsqueeze(-1))
        msk_loss = self.color_crit(output_rgb_msk, gt_rgb_msk)
        # scalar_stats.update({'msk_loss': msk_loss})
        loss += self.msk_loss_weight * msk_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats
