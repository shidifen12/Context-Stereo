import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from . import preprocess
import torch.nn.functional as F
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return Image.open(path)


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, datapath, loader=default_loader, dploader= disparity_loader):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.datapath = datapath

    def __getitem__(self, index):
        left  = os.path.join(self.datapath, self.left[index])
        right = os.path.join(self.datapath, self.right[index])
        disp_L= os.path.join(self.datapath, self.disp_L[index])

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)


        if self.training:  
           w, h = left_img.size
           th, tw = 256, 512
 
           x1 = random.randint(0, w - tw)
           y1 = random.randint(0, h - th)

           left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
           right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
           dataL = dataL[y1:y1 + th, x1:x1 + tw]

           processed = preprocess.get_transform(augment=False)  
           left_img   = processed(left_img)
           right_img  = processed(right_img)

           return left_img, right_img, dataL
        else:
        #    w, h = left_img.size

        #    left_img = left_img.crop((w-1232, h-368, w, h))
        #    right_img = right_img.crop((w-1232, h-368, w, h))
        #    w1, h1 = left_img.size

        #    dataL = dataL.crop((w-1232, h-368, w, h))
        #    dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256

        #    processed = preprocess.get_transform(augment=False)  
        #    left_img       = processed(left_img)
        #    right_img      = processed(right_img)

        #    return left_img, right_img, dataL

            # w, h = left_img.size

            # processed = preprocess.get_transform(augment=False)
            # left_img = torch.from_numpy(processed(left_img).numpy())
            # right_img = torch.from_numpy(processed(right_img).numpy())

            # # Pad to size 1248x384
            # top_pad = 384 - h
            # right_pad = 1248 - w
            # assert top_pad > 0 and right_pad > 0

            # # Pad images using F.pad
            # left_img = F.pad(left_img, (0, right_pad, top_pad, 0), mode='constant', value=0)
            # right_img = F.pad(right_img, (0, right_pad, top_pad, 0), mode='constant', value=0)

            # # Pad dataL
            # dataL = np.array(dataL)
            # dataL = dataL.astype(np.float32)
            # dataL = torch.from_numpy(dataL)
            # dataL = F.pad(dataL, (0, right_pad, top_pad, 0), mode='constant', value=0)
            # dataL = dataL.contiguous().float() / 256.0
            
            w, h = left_img.size

            # normalize
            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()

            # pad to size 1248x384
            top_pad = 384 - h
            right_pad = 1248 - w
            assert top_pad > 0 and right_pad > 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)

            dataL = np.lib.pad(dataL, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
            
            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)



