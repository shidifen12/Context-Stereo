import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines
from . import flow_transforms
import torchvision
import cv2
import copy

class SOTADataset(Dataset):
    def __init__(self, datapath_12, datapath_15, list_filename, training):
        self.datapath_15 = datapath_15
        self.datapath_12 = datapath_12
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):

        left_name = self.left_filenames[index].split('/')[1]
        if left_name.startswith('image'):
            self.datapath = self.datapath_15
        else:
            self.datapath = self.datapath_12
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))



        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        else:
            disparity = None





        # # 定义裁剪框的坐标和大小
        # left = 400  # 起始x坐标
        # top = 180   # 起始y坐标
        # right = 1600  # 结束x坐标
        # bottom = 550  # 结束y坐标

        # # 裁剪图像
        # left_img = left_img.crop((left, top, right, bottom))        
        # right_img = right_img.crop((left, top, right, bottom))   





        w, h = left_img.size     #输出HxW   1080X1920    ->  375, 1242
        # normalize
        processed = get_transform()

        left_img = processed(left_img).numpy()
        right_img = processed(right_img).numpy()


        # pad to size 1248x384
        top_pad = 384 - h
        right_pad = 1248 - w
        # top_pad = 768 - h
        # right_pad = 1376 - w

        assert top_pad > 0 and right_pad > 0
        # pad images
        left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
        # pad disparity gt
        if disparity is not None:
            assert len(disparity.shape) == 2
            disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

        if disparity is not None:
            return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]
                        }
        else:
            return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}
