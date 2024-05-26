import torch
from torch.nn import functional as F
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from typing import List
from lib.utils import data_config
from PIL import Image

def save_tensor_image(image_tensor, file_path):
    # Convert the tensor to a numpy array
    image_np = image_tensor.cpu().numpy()

    # Convert the numpy array to a PIL Image
    image_pil = Image.fromarray(np.uint8(image_np))

    # Save the PIL Image to the specified file path
    image_pil.save(file_path)


def unnormalize_img(img, mean, std):
    """
    img: [3, h, w]
    """
    img = img.detach().cpu().clone()
    # img = img / 255.
    img *= torch.tensor(std).view(3, 1, 1)
    img += torch.tensor(mean).view(3, 1, 1)
    min_v = torch.min(img)
    img = (img - min_v) / (torch.max(img) - min_v)
    return img

def pad_image(img: torch.Tensor, size: List[int], mode='constant', value=0.0):
    bs = img.shape[:-3]  # batch size
    img = img.reshape(-1, *img.shape[-3:])
    H, W = img.shape[-2:]  # H, W
    Ht, Wt = size
    pad = (0, Wt - W, 0, Ht - H)

    if Wt >= W and Ht >= H:
        img = F.pad(img, pad, mode, value)
    else:
        if Wt < W and Ht >= H:
            img = F.pad(img, (0, 0, 0, Ht - H), mode, value)
        if Wt >= W and Ht < H:
            img = F.pad(img, (0, Wt - W, 0, 0), mode, value)
        img = img[..., :Ht, :Wt]

    img = img.reshape(*bs, *img.shape[-3:])
    return img

def bgr_to_rgb(img):
    return img[:, :, [2, 1, 0]]


def horizon_concate(inp0, inp1):
    h0, w0 = inp0.shape[:2]
    h1, w1 = inp1.shape[:2]
    if inp0.ndim == 3:
        inp = np.zeros((max(h0, h1), w0 + w1, 3), dtype=inp0.dtype)
        inp[:h0, :w0, :] = inp0
        inp[:h1, w0:(w0 + w1), :] = inp1
    else:
        inp = np.zeros((max(h0, h1), w0 + w1), dtype=inp0.dtype)
        inp[:h0, :w0] = inp0
        inp[:h1, w0:(w0 + w1)] = inp1
    return inp


def vertical_concate(inp0, inp1):
    h0, w0 = inp0.shape[:2]
    h1, w1 = inp1.shape[:2]
    if inp0.ndim == 3:
        inp = np.zeros((h0 + h1, max(w0, w1), 3), dtype=inp0.dtype)
        inp[:h0, :w0, :] = inp0
        inp[h0:(h0 + h1), :w1, :] = inp1
    else:
        inp = np.zeros((h0 + h1, max(w0, w1)), dtype=inp0.dtype)
        inp[:h0, :w0] = inp0
        inp[h0:(h0 + h1), :w1] = inp1
    return inp


def transparent_cmap(cmap):
    """Copy colormap and set alpha values"""
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = 0.3
    return mycmap

cmap = transparent_cmap(plt.get_cmap('jet'))


def set_grid(ax, h, w, interval=8):
    ax.set_xticks(np.arange(0, w, interval))
    ax.set_yticks(np.arange(0, h, interval))
    ax.grid()
    ax.set_yticklabels([])
    ax.set_xticklabels([])


color_list = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.167, 0.000, 0.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000,
        0.50, 0.5, 0
    ]
).astype(np.float32)
colors = color_list.reshape((-1, 3)) * 255
colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)

def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi,ma]


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5