# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import ImageFilter, ImageOps
import math
import random

import torch
from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode
from typing import Tuple, List
from torch import Tensor


class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class JointCrop(transforms.RandomResizedCrop):
    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=InterpolationMode.BILINEAR,
    ):
        super().__init__(
            size=size, scale=scale, ratio=ratio, interpolation=interpolation
        )

    def get_params(
        self, img: Tensor, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = F.get_image_size(img)
        area = height * width
        log_ratio = torch.log(torch.tensor(ratio))
        flag1 = False
        flag2 = False

        log_scale = torch.log(torch.tensor(float(scale[1]) / float(scale[0]) - 0.1))

        # JointCrop 0 -> Uniform distribution
        log_scale_sample = torch.empty(1).uniform_(-log_scale, log_scale)

        scale_sample = torch.exp(log_scale_sample).item()

        target_area1 = area * torch.empty(1).uniform_(
            max(scale[0], scale[0] / scale_sample),
            min(scale[1] / scale_sample, scale[1]),
        )  # scale>1 -> the first item, scale<1 -> the second item
        target_area2 = target_area1 * scale_sample

        for _ in range(10):
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w1 = int(round(math.sqrt(target_area1 * aspect_ratio)))
            h1 = int(round(math.sqrt(target_area1 / aspect_ratio)))

            if 0 < w1 <= width and 0 < h1 <= height:
                i1 = torch.randint(0, height - h1 + 1, size=(1,)).item()
                j1 = torch.randint(0, width - w1 + 1, size=(1,)).item()
                flag1 = True
                break

        for _ in range(10):
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w2 = int(round(math.sqrt(target_area2 * aspect_ratio)))
            h2 = int(round(math.sqrt(target_area2 / aspect_ratio)))

            if 0 < w2 <= width and 0 < h2 <= height:
                i2 = torch.randint(0, height - h2 + 1, size=(1,)).item()
                j2 = torch.randint(0, width - w2 + 1, size=(1,)).item()
                flag2 = True
                break

        if flag1 and flag2:
            return i1, j1, h1, w1, i2, j2, h2, w2
        elif flag1:
            w = width
            h = height
            i = (height - h) // 2
            j = (width - w) // 2
            return i1, j1, h1, w1, i, j, h, w
        elif flag2:
            w = width
            h = height
            i = (height - h) // 2
            j = (width - w) // 2
            return i, j, h, w, i2, j2, h2, w2
        else:
            w = width
            h = height
            i = (height - h) // 2
            j = (width - w) // 2
            return i, j, h, w, i, j, h, w

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """

        i1, j1, h1, w1, i2, j2, h2, w2 = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(
            img, i1, j1, h1, w1, self.size, self.interpolation
        ), F.resized_crop(img, i2, j2, h2, w2, self.size, self.interpolation)


class TwoCropsTransform_JointCrop:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2, scale):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2
        self.base_transform = JointCrop(size=224, scale=scale)

    def __call__(self, x):
        im1, im2 = self.base_transform(x)
        im1 = self.base_transform1(im1)
        im2 = self.base_transform2(im2)
        return [im1, im2]


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)
