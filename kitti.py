import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
import random
import numpy as np
from dataloader import KITTILoader as DA
import utils.logger as logger
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import models.Context_Stereo

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='fintune on KITTI')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.25, 0.5, 1.])
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default='/home/admin1/WQL/FBGnet/kitti', help='datapath')
parser.add_argument('--epochs', type=int, default=600,
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=6,
                    help='batch size for training (default: 6)')
parser.add_argument('--test_bsize', type=int, default=8,
                    help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='kitti_results/t0',
                    help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default=None,
                    help='resume path')
parser.add_argument('--with_loss', type=int, default=0)
parser.add_argument('--pretrained', type=str, default='results/t0/checkpoint_28.tar',
                    help='pretrained model path')
parser.add_argument('--model', type=str, default=None, help='select model')
parser.add_argument('--loca', type=str, default=None, help='select log location')
parser.add_argument('--location', type=str, default=None, help='select loss image location')
parser.add_argument('--local', type=int, default=0, help='select label location')
parser.add_argument('--stage', type=int, default=1, help='stage of out')
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate (default: 5e-4)')



args = parser.parse_args()

if args.datatype == '2015':
    from dataloader import KITTIloader2015 as ls
elif args.datatype == '2012':
    from dataloader import KITTIloader12_15 as ls
elif args.datatype == 'other':
    from dataloader import diy_dataset as ls


def main():
    global args
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)
    
    log = logger.setup_logger(args.save_path +'/train.log')

    # train_list_filename = "filenames/kitti12_15_15_train.txt"
    train_list_filename = "filenames/kitti12_15_15_train.txt"
    val_list_filename = "filenames/kitti15_val.txt"

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(train_list_filename, val_list_filename)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_left_img, train_right_img, train_left_disp,True, datapath = args.datapath),
        batch_size=args.train_bsize, shuffle=True, num_workers=16, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False, datapath = args.datapath),
        batch_size=args.test_bsize, shuffle=False, num_workers=16, drop_last=False)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    # model = models.FBGnet.Net()
    # model = models.two.anynet_t3.AnyNet()
    model = models.Stereo(args.maxdisp)
    # model.cuda()
    model = nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999,))

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            log.info("=> loaded pretrained model '{}'"
                     .format(args.pretrained))
        else:
            log.info("=> no pretrained model found at '{}'".format(args.pretrained))
            log.info("=> Will start from scratch.")
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch']
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))

        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('Not Resume')
    
    cudnn.benchmark = True
    start_full_time = time.time()
    Test_loss = []
    best_d1 = 100
    best_e1 = 100
    for epoch in range(args.start_epoch, args.epochs):
        epoch = epoch + 1
        log.info('This is {}-th epoch'.format(epoch))

        adjust_learning_rate_D(optimizer, epoch)
        test_loss = train(TrainImgLoader, model, optimizer, log, epoch, lr_scheduler=None)
        Test_loss.append(test_loss)

        savefilename = args.save_path + '/checkpoint.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, savefilename)
        
        
        torch.cuda.empty_cache()
        if  epoch <= 300  and epoch>= 200:

            d1, e1 = test(TestImgLoader, model, log)
            if d1 < best_d1:
                best_d1 = d1
                torch.save({
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            }, "{}/best_result2_3_d.tar".format(args.save_path))
            if e1 < best_e1:
                best_e1 = e1
                torch.save({
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            }, "{}/best_result2_3_e.tar".format(args.save_path))
            torch.cuda.empty_cache() 
            
        elif  epoch<=400  and epoch >300:
            d1, e1 = test(TestImgLoader, model, log)
            if d1 < best_d1:
                best_d1 = d1
                torch.save({
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            }, "{}/best_result3_4_d.tar".format(args.save_path))
            if e1 < best_e1:
                best_e1 = e1
                torch.save({
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            }, "{}/best_result3_4_e.tar".format(args.save_path))
            torch.cuda.empty_cache()
                
        elif  epoch<=500  and epoch >400:
            d1, e1 = test(TestImgLoader, model, log)
            if d1 < best_d1:
                best_d1 = d1
                torch.save({
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            }, "{}/best_result4_5_d.tar".format(args.save_path))
            if e1 < best_e1:
                best_e1 = e1
                torch.save({
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            }, "{}/best_result4_5_e.tar".format(args.save_path))
            torch.cuda.empty_cache()
                
        elif  epoch<=600  and epoch >500:
            d1, e1 = test(TestImgLoader, model, log)
            if d1 < best_d1:
                best_d1 = d1
                torch.save({
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            }, "{}/best_result5_6_d.tar".format(args.save_path))
            if e1 < best_e1:
                best_e1 = e1
                torch.save({
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            }, "{}/best_result5_6_e.tar".format(args.save_path))
            torch.cuda.empty_cache()
    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))
    Loss_curve(args.epochs, Test_loss, location=args.location)



def train(dataloader, model, optimizer, log, epoch=0, lr_scheduler=None):
    stages = args.stage + args.with_loss
    losses = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)
    # Test_loss= []
    model.train()
    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        optimizer.zero_grad()
        mask = disp_L > 0
        mask.detach_()
        outputs = model(imgL, imgR)

        # num_out = len(outputs)
        num_out = stages
        outputs = [torch.squeeze(output, 1) for output in outputs]
        loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], reduction='mean')
                for x in range(num_out)]
        sum(loss).backward()
        optimizer.step()

        for idx in range(num_out):
            losses[idx].update(loss[idx].item())

        # if batch_idx % args.print_freq:
        info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(num_out)]
        info_str = '\t'.join(info_str)

        log.info('Epoch{} [{}/{}] {}'.format(
            epoch, batch_idx, length_loader, info_str))

    info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(stages)])
    log.info('Average train loss = ' + info_str)
    return losses[-1].avg




def test(dataloader, model, log,):

    stages = args.stage 
    D1s = [AverageMeter() for _ in range(stages)]
    pixel3 = [AverageMeter() for _ in range(stages)]
    EPE = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.eval()
    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        with torch.no_grad():
            outputs = model(imgL, imgR)
            for x in range(len(outputs)):
                output = torch.squeeze(outputs[x], 1)
                D1s[x].update(error_estimating(output, disp_L).item())
                pixel3[x].update(Thres_metric(output, disp_L).item())
                EPE[x].update(EPE_metric(output, disp_L).item())

        info_str0 = '\t'.join(['Stage {} = {:.4f}({:.4f})'.format(x, D1s[x].val, D1s[x].avg) for x in range(stages)])
        log.info('[{}/{}] {}'.format(
            batch_idx, length_loader, info_str0))


    info_str0 = ', '.join(['Stage {}={:.4f}'.format(x, D1s[x].avg) for x in range(stages)])
    info_str1 = ', '.join(['Stage {}={:.4f}'.format(x, pixel3[x].avg) for x in range(stages)])
    info_str2= ', '.join(['Stage {}={:.4f}'.format(x, EPE[x].avg) for x in range(stages)])


    log.info('D1 = ' + info_str0)
    log.info('Average test 3-Pixel Error = ' + info_str1)
    log.info('EPE = ' + info_str2)
    return D1s[-1].avg, EPE[-1].avg



def error_estimating(disp, ground_truth, maxdisp=192):    #D1-all
    gt = ground_truth
    mask = gt > 0
    mask = mask * (gt < maxdisp)

    errmap = torch.abs(disp - gt)
    err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()
    return err3.float() / mask.sum().float()

def Thres_metric(disp, ground_truth, thres=3.0, maxdisp = 192):  #test-3-pixel
 
    assert isinstance(thres, (int, float))
    gt = ground_truth
    mask = gt > 0
    mask = mask * (gt < maxdisp)
    errmap = torch.abs(disp - gt)[mask]
    err_mask = errmap > thres
    return torch.mean(err_mask.float())


def EPE_metric(disp, ground_truth, maxdisp=192):    #EPE
    gt = ground_truth
    mask = gt > 0
    mask = mask * (gt < maxdisp)
    D_est, D_gt = disp[mask], ground_truth[mask]
    return F.l1_loss(D_est, D_gt, reduction='mean')


def adjust_learning_rate_D(optimizer, epoch):
    if epoch <= 300:
        lr = args.lr
    elif epoch <= 600:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def update_learning_rate(log, optimizers, scheduler):
    scheduler.step()
    lr = optimizers.param_groups[0]['lr']


def Loss_curve(x_a, y_a, location):
    yl = len(y_a)
    x_s = x_a - yl
    int_v = range(x_s+1, x_a+1)
    fig, ax = plt.subplots()
    ax.plot(int_v, y_a, linewidth=0.8)
    plt.savefig(location, bbox_inches="tight")


if __name__ == '__main__':
    main()
