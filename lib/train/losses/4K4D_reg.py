import torch
import torch.nn as nn

from lib.config import cfg
from lib.train.losses.vgg_perceptual_loss import VGGPerceptualLoss

class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.color_crit = nn.MSELoss(reduction='mean')
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

        self.perc_loss_weight = cfg.train.lmbda1
        self.msk_loss_weight = cfg.train.lmbda2
        self.lpips = VGGPerceptualLoss()

    def mIoU_loss(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the mean intersection of union loss over masked regions
        x, y: B, N, 1
        """
        I = (x * y).sum(-1).sum(-1)
        U = (x + y).sum(-1).sum(-1) - I
        mIoU = (I / U.detach()).mean()
        return 1 - mIoU

    def compute_mask_loss(self, output, batch_msk):
        msk_loss = self.mIoU_loss(output.acc_map, batch_msk)
        loss = self.msk_loss_weight * msk_loss

        return loss

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

        msk_loss = self.compute_mask_loss(output['acc_map'], batch['msk'])
        scalar_stats.update({'msk_loss': msk_loss})
        loss += self.msk_loss_weight * msk_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats
