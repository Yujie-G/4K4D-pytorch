import torch
import torch.nn as nn

from lib.config import cfg
from lpips import LPIPS

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

        scalar_stats = {}
        loss = 0
        color_loss = self.color_crit(output['rgb'], batch['rgb'])
        scalar_stats.update({'color_mse': color_loss})
        loss += color_loss

        lpips_loss = self.lpips(output['rgb'], batch['rgb'])
        scalar_stats.update({'lpips': lpips_loss})
        loss += self.perc_loss_weight * lpips_loss

        msk_loss = self.color_crit(output['mask'], batch['mask'])
        scalar_stats.update({'msk_loss': msk_loss})
        loss += self.msk_loss_weight * msk_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats
