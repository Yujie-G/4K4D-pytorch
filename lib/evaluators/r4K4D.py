import numpy as np
from lib.config import cfg
import os
import json
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import imageio
from lib.utils import img_utils



class Evaluator:

    def __init__(self,):
        self.psnrs = []
        os.system('mkdir -p ' + cfg.result_dir)
        os.system('mkdir -p ' + cfg.result_dir + '/vis')

    def evaluate(self, output, batch, epoch=None):
        # assert image number = 1
        H, W = batch['meta']['H'].item(), batch['meta']['W'].item()
        pred_rgb = output['rgb'].reshape(H, W, 3).detach().cpu().numpy()
        gt_rgb = batch['rgb'].reshape(H, W, 3).detach().cpu().numpy()
        psnr_item = psnr(gt_rgb, pred_rgb, data_range=1.)
        self.psnrs.append(psnr_item)
        save_path = os.path.join(cfg.result_dir, f'vis/res-{epoch}.jpg')
        image_float64 = img_utils.horizon_concate(gt_rgb, pred_rgb) * 255.0
        image_int8 = image_float64.astype(np.uint8)
        imageio.imwrite(save_path, image_int8)

        pred_depth = output['dpt'].squeeze().detach().cpu().numpy()
        img_utils.save_numpy_image(pred_depth, os.path.join(cfg.result_dir, f'vis/depth-{epoch}.png'))
        # pred_depth = output['acc'].reshape(H, W, 1).detach().cpu().numpy()
        # img_utils.save_numpy_image(pred_depth, os.path.join(cfg.result_dir, f'vis/acc-{epoch}.png'))


    def summarize(self):
        ret = {}
        ret.update({'psnr': np.mean(self.psnrs)})
        print(ret)
        self.psnrs = []
        print('Save visualization results to {}'.format(cfg.result_dir))
        json.dump(ret, open(os.path.join(cfg.result_dir, 'metrics.json'), 'w'))
        return ret
