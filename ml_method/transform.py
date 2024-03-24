import numpy as np
from skimage.transform import rescale, rotate
from torchvision.transforms import Compose
from torchvision import transforms
import torch
import cv2
import torchvision.transforms.functional as TF
import random

from utilities.augmentation_utils import AugmentationUtils


def get_image_and_mask_transforms(scale=None, angle=None):
    transform_list = [RandomShiftWithMask(),
                      NormalizeImage(),
                      Resize((256, 256)),
                      ToTensor(),
                      DuplicateChannels()]

    if scale is not None:
        transform_list.append(Scale(scale))
    if angle is not None:
        transform_list.append(Rotate(angle))

    return Compose(transform_list)


def get_image_and_mask_transforms_inference():
    transform_list = [NormalizeImage(),
                      Resize((256, 256)),
                      ToTensor(),
                      DuplicateChannels()]
    return Compose(transform_list)


class DuplicateChannels(object):
    def __call__(self, sample):
        img, mask = sample
        # Duplicate the grayscale image along the channel dimension
        return torch.cat([img, img, img], dim=0), mask


class ToPIL(object):
    def __call__(self, sample):
        img, mask = sample
        transform = transforms.ToPILImage()
        return transform(img), transform(mask)


class ToTensor(object):
    def __call__(self, sample):
        img, mask = sample
        transform = transforms.ToTensor()
        img = transform(img)
        mask = transform(mask)
        return img, mask


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, mask = sample
        img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=self.size, interpolation=cv2.INTER_LINEAR)
        return img, mask


class NormalizeImage(object):
    def __call__(self, sample):
        img, mask = sample
        m, s = np.mean(img, axis=(0, 1)), np.std(img, axis=(0, 1))
        return (img - m) / s, mask


class Scale(object):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        image, mask = sample

        img_size = image.shape[0]

        scale = np.random.uniform(low=1.0 - self.scale, high=1.0 + self.scale)

        image = rescale(
            image,
            (scale, scale),
            multichannel=True,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )
        mask = rescale(
            mask,
            (scale, scale),
            order=0,
            multichannel=True,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )

        if scale < 1.0:
            diff = (img_size - image.shape[0]) / 2.0
            padding = ((int(np.floor(diff)), int(np.ceil(diff))),) * 2 + ((0, 0),)
            image = np.pad(image, padding, mode="constant", constant_values=0)
            mask = np.pad(mask, padding, mode="constant", constant_values=0)
        else:
            x_min = (image.shape[0] - img_size) // 2
            x_max = x_min + img_size
            image = image[x_min:x_max, x_min:x_max, ...]
            mask = mask[x_min:x_max, x_min:x_max, ...]

        return image, mask


class Rotate(object):

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image, mask = sample

        angle = np.random.uniform(low=-self.angle, high=self.angle)
        image = rotate(image, angle, resize=False, preserve_range=True, mode="constant")
        mask = rotate(
            mask, angle, resize=False, order=0, preserve_range=True, mode="constant"
        )
        return image, mask


class HorizontalFlip(object):

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.flip_prob:
            return image, mask

        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

        return image, mask


class RandomShiftWithMask(object):
    def __init__(self, translate=0.1):
        self.translate = translate

    def __call__(self, sample):
        img, mask = sample
        height, width = img.shape
        shift_h = int(self.translate * height)
        shift_w = int(self.translate * width)
        translate_h = np.random.randint(-shift_h, shift_h)
        translate_w = np.random.randint(-shift_w, shift_w)
        img = AugmentationUtils.shift_image(img, w=translate_w, h=translate_h)

        # Apply affine transformation to mask
        mask = AugmentationUtils.shift_image(mask, w=translate_w, h=translate_h)

        return img, mask
