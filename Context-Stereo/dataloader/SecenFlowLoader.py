import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
from . import preprocess
from . import listflowfile as lt
from . import readpfm as rp
import numpy as np

from datasets.data_io import get_transform, read_all_lines, pfm_imread
from datasets import flow_transforms
import torchvision
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return rp.readPFM(path)


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL, scaleL = self.dploader(disp_L)
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)

        if self.training:
            # w, h = left_img.size
            # th, tw = 256, 512

            # x1 = random.randint(0, w - tw)
            # y1 = random.randint(0, h - th)

            # left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            # right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            # dataL = dataL[y1:y1 + th, x1:x1 + tw]

            # processed = preprocess.get_transform(augment=False)
            # left_img = processed(left_img)
            # right_img = processed(right_img)

            th, tw = 256, 512
            random_brightness = np.random.uniform(0.5, 2.0, 2)
            random_gamma = np.random.uniform(0.8, 1.2, 2)
            random_contrast = np.random.uniform(0.8, 1.2, 2)
            random_saturation = np.random.uniform(0, 1.4, 2)

            left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness[0])
            right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness[1])

            left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma[0])
            right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma[1])

            left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast[0])
            right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast[1])

            left_img = torchvision.transforms.functional.adjust_saturation(left_img, random_saturation[0])
            right_img = torchvision.transforms.functional.adjust_saturation(right_img, random_saturation[1])

            right_img = np.array(right_img)
            left_img = np.array(left_img)

            # geometric unsymmetric-augmentation
            angle = 0;
            px = 0
            if np.random.binomial(1, 0.5):
                angle = 0.05
                px = 1
            co_transform = flow_transforms.Compose([
                flow_transforms.RandomCrop((th, tw)),
            ])
            augmented, dataL = co_transform([left_img, right_img], dataL)
            left_img = augmented[0]
            right_img = augmented[1]

            # randomly occlude a region
            right_img.flags.writeable = True
            if np.random.binomial(1,0.5):
              sx = int(np.random.uniform(35,100))
              sy = int(np.random.uniform(25,75))
              cx = int(np.random.uniform(sx,right_img.shape[0]-sx))
              cy = int(np.random.uniform(sy,right_img.shape[1]-sy))
              right_img[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(right_img,0),0)[np.newaxis,np.newaxis]

            # w, h = left_img.size

            dataL = np.ascontiguousarray(dataL, dtype=np.float32)
            # disparity_low = cv2.resize(disparity, (tw//4, th//4), interpolation=cv2.INTER_NEAREST)

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)






            return left_img, right_img, dataL
        else:
            # w, h = left_img.size
            # th, tw = 512, 960

            # x1 = random.randint(0, w - tw)
            # y1 = random.randint(0, h - th)

            # left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            # right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            # dataL = dataL[y1:y1 + th, x1:x1 + tw]
            w, h = left_img.size
            left_img = left_img.crop((w - 960, h - 544, w, h))
            right_img = right_img.crop((w - 960, h - 544, w, h))
            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)


            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)




