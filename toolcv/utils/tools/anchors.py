import torch
import numpy as np
from math import ceil


def get_base_anchors(base_size=32, ratios=[0.5, 1.0, 2.0],
                     scales=[2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)]):
    """
    base_size * scales 得到 预设值的 size
    """
    ratios = torch.tensor(ratios)
    scales = torch.tensor(scales)
    x_center = 0
    y_center = 0

    w = base_size
    h = base_size
    # 计算高宽比例
    h_ratios = torch.sqrt(ratios)
    w_ratios = 1 / h_ratios
    # base_size 乘上宽高比例乘上尺度，就可以得到 n 个 anchor 的原图尺度wh值
    ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
    hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
    # 得到 x1y1x2y2 格式的 base_anchor 坐标值
    base_anchors = [
        x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
        y_center + 0.5 * hs
    ]
    # 堆叠起来即可
    base_anchors = torch.stack(base_anchors, dim=-1)

    return base_anchors


# 给定预设值anchor的大小
# [x1,y1,x2,y2] ,输入图像大小
def get_base_anchorsv2(input_size: list, wh: list):  # 给定预设值anchor的大小
    # wh 缩放到0~1
    h, w = input_size  # 输入的大小
    wh = np.array(wh)
    wh = wh * np.array([w, h])[None]
    ws = wh[:, 0]
    hs = wh[:, 1]
    base_anchors = np.stack([-ws, -hs, ws, hs], axis=1) / 2
    return torch.from_numpy(base_anchors).float()


def _meshgrid(x, y, row_major=True):
    """Generate mesh grid of x and y.

    Args:
        x (torch.Tensor): Grids of x dimension.
        y (torch.Tensor): Grids of y dimension.
        row_major (bool, optional): Whether to return y grids first.
            Defaults to True.
            等价于     # Y,X = torch.meshgrid(y,x)
                     # X, Y = np.meshgrid(x, y)

    Returns:
        tuple[torch.Tensor]: The mesh grids of x and y.
    """
    # use shape instead of len to keep tracing while exporting to onnx
    xx = x.repeat(y.shape[0])
    yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)
    if row_major:
        return xx, yy
    else:
        return yy, xx


def get_anchors(img_shape, device="cpu", strides=[], base_sizes=[32, 64, 128, 256, 512],
                ratios=[0.5, 1.0, 2.0], scales=[2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)], concat=True, clip=True,
                normal=True, wh=[], featmap_sizes=[]):
    """
    return: [m,4] x1,y1,x2,y2

    example:
        >>>from math import ceil

        >>>img_shape = (600, 800)
        >>>base_sizes = [32, 64, 128, 256, 512]
        >>>strides = [8, 16, 32, 64, 128]
        >>>anchors = get_anchors(img_shape, "cuda",strides, base_sizes)

        -------------------------------------------------------------------------------------
        >>>img_shape = (300, 300)
        >>>base_sizes = [1]
        >>>strides = [16]
        >>>scales = [64, 128, 256]
        >>>anchors = get_anchors(img_shape, "cuda", strides, base_sizes)

        ---------------------------------------------------------------------------------------
        >>>img_shape = (300, 300)
        >>>strides = [16]
        >>>anchors = get_anchors(img_shape, "cuda", strides, wh=[[[0.1, 0.2], [0.2, 0.3], [0.3, 0.5]]])

    """
    if len(featmap_sizes) == 0:
        featmap_sizes = [(ceil(img_shape[0] / s), ceil(img_shape[1] / s)) for s in strides]
    h, w = img_shape
    out = []
    nums = len(strides)
    for i in range(nums):
        if len(wh) > 0:
            base_anchors = get_base_anchorsv2(img_shape, wh[i]).to(device)
        else:
            base_size = base_sizes[i]
            base_anchors = get_base_anchors(base_size, ratios, scales).to(device)
        feat_h, feat_w = featmap_sizes[i]
        stride = strides[i]
        # 遍历特征图上所有位置，并且乘上 stride，从而变成原图坐标
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = _meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # (0,0) 位置的 base_anchor，假设原图上坐标 shifts，即可得到特征图上面每个点映射到原图坐标上的 anchor
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)

        if clip:
            all_anchors[..., [0, 2]] = all_anchors[..., [0, 2]].clamp(0, w - 1)
            all_anchors[..., [1, 3]] = all_anchors[..., [1, 3]].clamp(0, h - 1)
        if normal:
            all_anchors[..., [0, 2]] /= w
            all_anchors[..., [1, 3]] /= h

        out.append(all_anchors)

    if concat:
        return torch.cat(out, 0)

    return out


if __name__ == "__main__":
    img_shape = (300, 300)
    strides = [16]
    anchors = get_anchors(img_shape, "cuda", strides, wh=[[[0.1, 0.2], [0.2, 0.3], [0.3, 0.5]]],concat=False)

    print(anchors.shape)
    print(anchors.device)
