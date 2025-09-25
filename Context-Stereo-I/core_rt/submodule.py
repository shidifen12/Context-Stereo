import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x


class BasicConv_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
        super(BasicConv_IN, self).__init__()

        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x


class Conv2x_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, IN=True, relu=True, keep_dispc=False):
        super(Conv2x_IN, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv_IN(out_channels*2, out_channels*mul, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv_IN(out_channels, out_channels, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost

def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume
        



def norm_correlation(fea1, fea2):
    cost = torch.mean(((fea1/(torch.norm(fea1, 2, 1, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 1, True)+1e-05))), dim=1, keepdim=True)
    return cost

def build_norm_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = norm_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = norm_correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume

def correlation(fea1, fea2):
    cost = torch.sum((fea1 * fea2), dim=1, keepdim=True)
    return cost

def build_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume



def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, :] = refimg_fea[:, :, :, :]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume

def disparity_regression(prob, maxdisp, interval):
    assert len(prob.shape) == 4
    disp_values = torch.arange(0, maxdisp, interval, dtype=prob.dtype, device=prob.device)
    disp_values = disp_values.view(1, maxdisp//interval, 1, 1)
    return torch.sum(prob * disp_values, 1, keepdim=True)


class FeatureAtt(nn.Module):
    def __init__(self, cv_chan, feat_chan):
        super(FeatureAtt, self).__init__()

        self.feat_att = nn.Sequential(
            BasicConv(feat_chan, feat_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan//2, cv_chan, 1))

    def forward(self, cv, feat):
        '''
        '''
        feat_att = self.feat_att(feat).unsqueeze(2)
        cv = torch.sigmoid(feat_att)*cv
        return cv

def context_upsample(disp_low, up_weights):
    ###
    # cv (b,1,h,w)
    # sp (b,9,4*h,4*w)
    ###
    b, c, h, w = disp_low.shape       
    disp_unfold = F.unfold(disp_low.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)
    disp_unfold = F.interpolate(disp_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)
    disp = (disp_unfold*up_weights).sum(dim=1,keepdim=True)      
    return disp

def norm_correlation(fea1, fea2):
    cost = torch.mean(((fea1/(torch.norm(fea1, 2, 1, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 1, True)+1e-05))), dim=1, keepdim=True) #torch.norm:对feal求2范数，在dim维度，True：且保持那个维度
    return cost

def build_norm_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = norm_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = norm_correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume

class CPF(nn.Module):
    def __init__(self, in_channels, after_relu=False, with_channel=True):
        super(CPF, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        # self.guide = GuideNN(64)
        # self.slice = Slice()
        self.f_x = BasicConv(in_channels, in_channels, is_3d=True, kernel_size=3, padding=1, stride=1)                               
        self.f_y = BasicConv(in_channels, in_channels, is_3d=True, kernel_size=3, padding=1, stride=1)
        self.f_z = BasicConv(in_channels, in_channels, is_3d=True, kernel_size=3, padding=1, stride=1)
        
        if with_channel:
            self.up = BasicConv(in_channels, in_channels, is_3d=True, kernel_size=3, padding=1, stride=1)
            #self.up1 = BasicConv(in_channels, in_channels, is_3d=True, kernel_size=3, padding=1, stride=1)
            #self.up2 = BasicConv(in_channels, in_channels, is_3d=True, kernel_size=3, padding=1, stride=1)
        if after_relu:
            self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, y):
        # map = self.guide(img)
        # input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
        # print("y.size():{}".format(y.size()))    
        y_q = self.f_y(y)
        x_k = self.f_x(x)
        
        if self.with_channel:
            sim_map =self.up(x_k * y_q)
            #sim_map1 =self.up1(sim_map)
            #sim_map2 =self.up2(sim_map1)
            #sim_map =torch.sigmoid(sim_map + sim_map1 + sim_map2)
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
               
        z = (1-sim_map)*x + sim_map*y
        z =  self.f_z(z)
        z = x+y+z
        #z = torch.cat((x,z,y),dim=1)
        return z

class FeatureAtt(nn.Module):
    def __init__(self, cv_chan, feat_chan):
        super(FeatureAtt, self).__init__()

        self.feat_att = nn.Sequential(
            BasicConv(feat_chan, feat_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan//2, cv_chan, 1))

    def forward(self, cv, feat):
        feat_att = self.feat_att(feat).unsqueeze(2)
        cv = torch.sigmoid(feat_att)*cv
        return cv
    
class hourglass_fusion(nn.Module):
    def __init__(self, in_channels): #8
        super(hourglass_fusion, self).__init__()
        #编码器：下采样模块由一个stride=2的3*3*3三维卷积和一个stride=1的3*3*3三维卷积组成，下采样的几何特征尺寸为B*6C*D/32*H/32*W/32
        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,  #conv3d+bn+relu
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,#conv3d+bn+relu
                                             padding=1, stride=1, dilation=1))
        
        # self.conv2_ = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,#conv3d+bn+relu
        #                                      padding=1, stride=2, dilation=1),
        #                            BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,#conv3d+bn+relu
        #                                      padding=1, stride=1, dilation=1))                             
 
                                         
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,#conv3d+bn+relu
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,#conv3d+bn+relu
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,#conv3d+bn+relu
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,#conv3d+bn+relu
                                             padding=1, stride=1, dilation=1)) 

        #解码器由CGF和上采样模块来解码高分辨率几何特征，上采样模块:1个stride=2的4*4*4的3D转置卷积（用来上采样，分辨率翻倍）
        #和2个stride=1的3*3*3的3D卷积组成（融合解码器上采样后的特征图与解码器对应分辨率特征图contact的特征图）
        #conv3_up：一个4*4*4 ConvTransposed3d,agg_0:一个1*1*1 Conv3d和2个3*3*3 Conv3d
        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True, #ConvTranspose3d+bn+relu
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,#ConvTranspose3d+bn+relu
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, bn=False,#ConvTranspose3d+bn+relu
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),#conv3d+bn+relu
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))

        

              
        self.CPF_1 = CPF(in_channels*2)
        self.CPF_2 = CPF(in_channels*4)
        
        self.FA_16 = FeatureAtt(in_channels*4, 192)
        self.FA_8 = FeatureAtt(in_channels*2, 64)
        
        self.GSAM_8_d = GSAM(in_channels*2, 64, 1)
        self.GSAM_16_d = GSAM(in_channels*4, 192, 1)
        self.GSAM_32_d = GSAM(in_channels*6, 160, 1)
        # self.FA_16_up = FeatureAtt(in_channels*4, 192)
        # self.FA_8_up = FeatureAtt(in_channels*2, 64)
        
    def forward(self, x, imgs): #B*8*D/4*H/4*W/4,[96*H/4*W/4,64*H/8*W/8,192*H/16*W/16,160*H/32*W/32]
        #编码结构
        conv1 = self.conv1(x) #B*8*D/4*H/4*W/4->B*16*D/8*H/8*W/8
        conv1 = self.GSAM_8_d(conv1, imgs[1])
        
        conv2 = self.conv2(conv1) #B*16*D/8*H/8*W/8->B*32*D/16*H/16*W/16
        conv2 = self.GSAM_16_d(conv2, imgs[2])
        
        conv3 = self.conv3(conv2) #B*32*D/16*H/16*W/32->B*48*D/32*H/32*W/32
        #解码结构              
        conv3 = self.GSAM_32_d(conv3, imgs[3])
        conv3_up = self.conv3_up(conv3) #B*48*D/32*H/32*W/32->B*32*D/16*H/16*W/16
        conv2 = self.CPF_2(self.FA_16(conv3_up, imgs[2]), self.FA_16(conv2, imgs[2])) #B*64*D/16*H/16*W/16
        conv2 = self.agg_0(conv2) #B*64*D/16*H/16*W/16->B*32*D/16*H/16*W/16

        # conv2 = self.CCF_3(self.FA_16_up(conv2, imgs[2]))
        conv2_up = self.conv2_up(conv2) #B*32*D/16*H/16*W/16->B*16*D/8*H/8*W/8
        conv1 =  self.CPF_1(self.FA_8(conv2_up, imgs[1]), self.FA_8(conv1, imgs[1])) #B*32*D/8*H/8*W/8
        conv1 = self.agg_1(conv1) #B*32*D/8*H/8*W/8->B*16*D/8*H/8*W/8
        
        # conv1 = self.CCF_4(self.FA_8_up(conv1, imgs[1]))
        conv = self.conv1_up(conv1) #B*16*D/8*H/8*W/8->B*1*D/4*H/4*W/4

        return conv #B*1*D/4*H/4*W/4

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化

        # 两个卷积层用于从池化后的特征中学习注意力权重
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 第一个卷积层，降维
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 第二个卷积层，升维
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 对平均池化的特征进行处理
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 对最大池化的特征进行处理
        out = avg_out + max_out  # 将两种池化的特征加权和作为输出
        return self.sigmoid(out)  # 使用sigmoid激活函数计算注意力权重
    
class SpatialAttention(nn.Module): #空间注意力机制
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x

class GSAM(nn.Module):
    def __init__(self, in_channels, feat_channels, scale):  
        super(GSAM, self).__init__()   
        self.scale = scale    
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = BasicConv(in_channels, in_channels, is_3d=True, bn=False, relu=False, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv2 = BasicConv(in_channels, in_channels, is_3d=True, bn=True, relu=False, kernel_size=(1, 3, 3), padding=(0, 1, 1)) 
        self.conv3 = BasicConv(in_channels, in_channels, is_3d=True, bn=False, relu=False, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv4 = BasicConv(in_channels, in_channels, is_3d=True, bn=True, relu=False, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv5 = BasicConv(in_channels, in_channels, is_3d=True, bn=False, relu=False, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv6 = BasicConv(in_channels, in_channels, is_3d=True, bn=True, relu=False, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        
        # if self.scale > 0:
        #     self.DC = nn.ModuleList([
        #         BasicConv(
        #             in_channels, 
        #             in_channels, 
        #             is_3d=True,
        #             kernel_size=(3, 3, 3), 
        #             padding=i, 
        #             dilation=i
        #         ) for i in range(1, self.scale + 1)
        #     ])
        
        self.semantic = nn.Sequential(  
                        BasicConv(feat_channels, feat_channels // 2, kernel_size=1, stride=1, padding=0), 
                        nn.Conv2d(feat_channels // 2, in_channels, 1))
        self.Spatial_att = SpatialAttention(kernel_size=7)
        self.Channel_att = ChannelAttention(in_channels)
        # self.agg_2 = nn.Sequential(
        #                 BasicConv(in_channels, in_channels, is_3d=True, kernel_size=1, padding=0, stride=1)
        #                 )
        
    def forward(self, x, feat):   # B*8*D/8*H/8*W/8, 64*H/8*W/8
        feat = self.semantic(feat)  # B*64*H/8*W/8->B*8*H/8*W/8
        att = self.Spatial_att(feat).unsqueeze(2)
        att_1 = self.Channel_att(feat).unsqueeze(2)
        
        for i in range(self.scale):
            rem = x
            #x = self.DC[i](x)  # 根据 i 调用对应的 Conv3d 层
            x = x + feat.unsqueeze(2)
            x = self.conv2(self.conv1(x)) 
            x = self.relu(x + rem)
            
            x_ = att * x
            x_1 = att_1 * x
            x = x + self.conv4(self.conv3(x_)) + self.conv6(self.conv5(x_1))
            # x = self.relu(x)
            #x = x + rem
            # x = self.agg_2(x)
        return x


