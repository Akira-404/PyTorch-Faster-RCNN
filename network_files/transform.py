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
        self.min_size = min_size        # 最小边
        self.max_size = max_size        # 最大边
        self.image_mean = image_mean    # 均值
        self.image_std = image_std      # 方差

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

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox
        return image, target

    # _onnx_batch_images() is an implementation of
    # batch_images() that is supported by ONNX tracing.
    @torch.jit.unused
    def _onnx_batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        max_size = []
        for i in range(images[0].dim()):
            max_size_i = torch.max(torch.stack([img.shape[i] for img in images]).to(torch.float32)).to(torch.int64)
            max_size.append(max_size_i)
        stride = size_divisible
        max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size[2] = (torch.ceil((max_size[2].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size = tuple(max_size)

        # work around for
        # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        # which is not yet supported in onnx
        padded_imgs = []
        for img in images:
            padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
            padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
            padded_imgs.append(padded_img)

        return torch.stack(padded_imgs)

    # 计算一组list中的最大值
    def max_by_axis(self, the_list):
        """
        :param the_list: List[List[int]]
        :return: List[int]
        """
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images, size_divisible=32):
        """
        将图片打包成一个batch
        :param images:input images
        :param size_divisible: 将图片调整为size_divisible的整数倍，可能是为了计算机更好的计算
        :return: 一个打包好的tensor
        """

        if torchvision._is_tracing():
            # batch_images() does not export well to ONNX
            # call _onnx_batch_images() instead
            return self._onnx_batch_images(images, size_divisible)

        max_size = self.max_by_axis([list(img.shape) for img in images])

        stride = float(size_divisible)
        # 将height向上调整到stride的整数倍
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        # 将width向上调整到stride的整数倍
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_size = [len(images)] + max_size

        batched_imgs = images[0].new_full(batch_size, 0)
        for img, pad_img in zip(images, batched_imgs):
            # 将输入images中的每张图片复制到新的batched_imgs的每张图片中，对齐左上角，保证bboxes的坐标不变
            # 这样保证输入到网络中一个batch的每张图片的shape相同
            # copy_: Copies the elements from src into self tensor and returns self
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(self, result, image_shapes, original_image_sizes):
        """
        将bboxes还原到原图片尺度上
        :param result:              list[dict[str,tensor]]
        :param image_shapes:        list[tuple[int,int]]
        :param original_image_sizes:list[tuple[int,int]]
        """
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            return result

    def __repr__(self):
        """自定义输出实例化对象的信息，可通过print打印实例信息"""
        format_string = self.__class__.__name__ + '('
        _indent = '\n    '
        format_string += "{0}Normalize(mean={1}, std={2})".format(_indent, self.image_mean, self.image_std)
        format_string += "{0}Resize(min_size={1}, max_size={2}, mode='bilinear')".format(_indent, self.min_size,
                                                                                         self.max_size)
        format_string += '\n)'
        return format_string

    def forward(self, images, targets=None):
        """
        :param images:list[tensor]
        :param targets:Optional[List[Dict[str, Tensor]]]:
        :return:
        """
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None
            if image.dim != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            # 对图片进行标准化
            image = self.normalize(image)
            # 对图片和bbox进行缩放
            image, target_index = self.resize(image, target_index)
            images[i] = image

            if targets is not None and target_index is not None:
                targets[i] = target_index

        # 记录resize后的尺寸
        image_sizes = [img.shape[-2:] for img in images]
        # 将图片打包j
        images = self.batch_images(images)
        image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])

        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)

        return image_list, targets


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
