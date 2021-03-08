import math
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn, Tensor
import torchvision

from network_files.image_list import ImageList


class GeneralizedRCNNTransform(nn.Module):
    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size  # 最小边
        self.max_size = max_size  # 最大边
        self.image_mean = image_mean  # 均值
        self.image_std = image_std  # 方差

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)

        return (image - mean[:, None, None]) / std[:, None, None]

    def torch_choice(self, k):
        # type: (List[int]) -> int
        """
        Implements `random.choice` via torch ops so it can be compiled with
        TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
        is fixed.
        """
        index = int(torch.empty(1).uniform_(0., float(len(k))).item())
        return k[index]

    def resize(self, image, target):
        """
        :param image:input images
        :param target: image info(bboxes)
        :return:image:resize image
        :return: target resize image info
        """
        # image shape=[channel,height,weith]
        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])

        # 获取图片中最小最大边
        min_size = float(torch.min(im_shape))
        max_size = float(torch.max(im_shape))

        if self.training:
            size = float(self.torch_choice(self.min_size))
        else:
            size = float(self.min_size[-1])
        # 根据指定最小边长和图片最小边长计算缩放比例
        scale_factor = size / min_size

        # 如果使用该缩放比例计算的图片最大边长大于指定的最大边长
        if max_size * scale_factor > max_size:
            scale_factor = self.max_size / max_size

        # 利用插值算法缩放图片
        image = torch.nn.functional.interpolate(
            image[None],
            scale_factor=scale_factor,
            mode='bilinear',
            recompute_scale_factor=True,
            align_corners=False)[0]

        if target is None:
            return image, target


def resize_boxes(boxes, original_size, new_size):
    """
    :param boxes:tensor
    :param original_size:list[int],原始尺寸
    :param new_size: list[int],新尺寸
    :return:tensor->[]
    """

    # ratios = []
    # for s, s_orig in zip(new_size, original_size):
    #     ratios.append(torch.tensor(s, dtype=torch.float32, device=boxes.device) /
    #                   torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
    #                   )

    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]

    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_width
    ymax = ymax * ratio_width

    return torch.stack((xmin, ymin, xmax, ymax), dim=1)
