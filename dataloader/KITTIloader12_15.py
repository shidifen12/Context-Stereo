import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(train_list_filename, val_list_filename):


    train_lines = read_all_lines(train_list_filename)
    train_splits = [line.split() for line in train_lines]
    val_lines = read_all_lines(val_list_filename)
    val_splits = [line.split() for line in val_lines]

    left_train  = [x[0] for x in train_splits]
    right_train = [x[1] for x in train_splits]
    disp_train_L = [x[2] for x in train_splits]
    left_val = [x[0] for x in val_splits]
    right_val = [x[1] for x in val_splits]
    disp_val_L = [x[2] for x in val_splits]

    return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L
