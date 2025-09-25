import time
import torch
import argparse
import sys
import thop
from easydict import EasyDict
from tqdm import tqdm
from models.Context_Stereo import Context_Stereo
import torch.nn as nn

from torchinfo import summary



def main():
    model = Context_Stereo(192)
    model.cuda()
    shape = [1, 3, 384, 1248]
    left = torch.randn(shape).cuda()
    right = torch.randn(shape).cuda()
    summary(model, input_data=(left, right), col_names=["input_size", "output_size", "num_params", "mult_adds"])    
    infer_time(model, left, right)

@torch.no_grad()
def infer_time(model, left, right):
    repetitions = 100
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(10):
            outputs = model(left, right)
    all_time = 0
    print('testing ...\n')
    with torch.no_grad():
        for _ in tqdm(range(repetitions)):
            infer_start = time.perf_counter()
            result = model(left,right)
            torch.cuda.synchronize()
            all_time += time.perf_counter() - infer_start
    print(all_time / repetitions * 1000)


if __name__ == '__main__':
    main()
