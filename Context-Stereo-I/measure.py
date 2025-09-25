# @Time    : 2024/3/1 11:17
# @Author  : zhangchenming
import time
import torch
import argparse
import sys
import thop
from easydict import EasyDict
from tqdm import tqdm
import os
from core_rt.rt_igev_stereo import IGEVStereo, autocast

#sys.path.insert(0, './')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))


# def parse_config():
#     parser = argparse.ArgumentParser(description='arg parser')
#     parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
#     #D:\\cocunment\\OpenStereo-2\\cfgs\\lightstereo\\lightstereo_m_kitti.yaml
#     parser.add_argument('--cfg_file', type=str, default="/home/wei/yxw/Test_speed/OpenStereo/cfgs/lightstereo/lightstereo_l_kitti.yaml", help='specify the config for training')

#     args = parser.parse_args()
#     # yaml_config = common_utils.config_loader(args.cfg_file)
#     # cfgs = EasyDict(yaml_config)
#     args.run_mode = 'measure'
#     return args, cfgs


def main(args):
    
    model = IGEVStereo(args)
    #model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.cuda()
    model.eval()
    shape = [1, 3, 384, 1248]
    infer_time(model, shape, args)
    #measure(model, shape)


@torch.no_grad()
def measure(model, shape):
    model.eval()

    left = torch.randn(shape).cuda()
    right = torch.randn(shape).cuda()

    flops, params = thop.profile(model, inputs=(left,right))
    print("Number of calculates:%.2fGFlops" % (flops / 1e9))
    print("Number of parameters:%.2fM" % (params / 1e6))


@torch.no_grad()
def infer_time(model, shape, args):
    model.eval()
    repetitions = 100

    left = torch.randn(shape).cuda()
    right = torch.randn(shape).cuda()

    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(10):
            _ = model(left,right)

    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    # torch.cuda.synchronize()

    # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    # timings = np.zeros((repetitions, 1))

    all_time = 0
    print('testing ...\n')
    with torch.no_grad():
        for _ in tqdm(range(repetitions)):
            # starter.record()
            infer_start = time.perf_counter()
            # infer_start = time.time()
            result = model(left,right)
            #print(result.keys())
            # ender.record()
            torch.cuda.synchronize() 
            all_time += time.perf_counter() - infer_start
            #torch.cuda.synchronize()  # 等待GPU任务完成

            # curr_time = starter.elapsed_time(ender)  # 从 starter 到 ender 之间用时,单位为毫秒
            # timings[rep] = curr_time

    # avg = timings.sum() / repetitions
    # print('\navg_time=%.3fms\n' % avg)
    print(all_time / repetitions * 1000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/igev_rt/sceneflow.pth')
    parser.add_argument('--dataset', help="dataset for evaluation", default='sceneflow', choices=["eth3d", "kitti", "sceneflow"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=4, help='number of flow-field updates during forward pass')

    # Architecure choices
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=96, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp range")
    args = parser.parse_args()

    
    main(args)
