from __future__ import print_function
from collections import Counter
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from .submodule import *
import math
import gc
import time
import timm


class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):  
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Feature(SubModule): 
    def __init__(self):
        super(Feature, self).__init__()
        pretrained =True
        model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True, pretrained_cfg_overlay=dict(file='mobilenetv2_100_ra-b33bc2c4.pth'))
        layers = [1,2,3,5,6]
        chans = [16, 24, 32, 96, 160]
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])    
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])  
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])  
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])   
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])   

        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)   

    def forward(self, x): 
        x = self.bn1(self.conv_stem(x)) # 
        x2 = self.block0(x)  
        x4 = self.block1(x2)  
        x8 = self.block2(x4)    
        x16 = self.block3(x8)  
        x32 = self.block4(x16)  
        return [x4, x8, x16, x32] 

class FeatUp(SubModule):
    def __init__(self):
        super(FeatUp, self).__init__()
        chans = [16, 24, 32, 96, 160]
        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)  
        self.deconv16_8 = Conv2x(chans[3]*2, chans[2], deconv=True, concat=True) 
        self.deconv8_4 = Conv2x(chans[2]*2, chans[1], deconv=True, concat=True) 
        self.conv4 = BasicConv(chans[1]*2, chans[1]*2, kernel_size=3, stride=1, padding=1)  
        self.weight_init()

    def forward(self, featL, featR=None):
        x4, x8, x16, x32 = featL  

        y4, y8, y16, y32 = featR  
        x16 = self.deconv32_16(x32, x16)  
        y16 = self.deconv32_16(y32, y16)  
        
        x8 = self.deconv16_8(x16, x8)  
        y8 = self.deconv16_8(y16, y8)  
        
        x4 = self.deconv8_4(x8, x4)  
        y4 = self.deconv8_4(y8, y4)  
        
        x4 = self.conv4(x4)  
        y4 = self.conv4(y4)  

        return [x4, x8, x16, x32], [y4, y8, y16, y32]  
    
class AGS(nn.Module):
    def __init__(self, in_channels, after_relu=False, with_channel=True):
        super(AGS, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = BasicConv(in_channels, in_channels, is_3d=True, kernel_size=3, padding=1, stride=1)                               
        self.f_y = BasicConv(in_channels, in_channels, is_3d=True, kernel_size=3, padding=1, stride=1)
        self.f_z = BasicConv(in_channels, in_channels, is_3d=True, kernel_size=3, padding=1, stride=1)
        
        if with_channel:
            self.up = BasicConv(in_channels, in_channels, is_3d=True, kernel_size=3, padding=1, stride=1)
        if after_relu:
            self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, y):
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
        y_q = self.f_y(y)
        x_k = self.f_x(x)
        
        if self.with_channel:
            sim_map =self.up(x_k * y_q)
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
               
        z = (1-sim_map)*x + sim_map*y
        z =  self.f_z(z)
        z = x+y+z
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
    def __init__(self, in_channels): 
        super(hourglass_fusion, self).__init__()
        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,   
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3, 
                                             padding=1, stride=1, dilation=1))
                                               
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3, 
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3, 
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3, 
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3, 
                                             padding=1, stride=1, dilation=1)) 

        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True, 
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True, 
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 1, deconv=True, is_3d=True, bn=False, 
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1), 
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))
       
        self.CPF_1 = AGS(in_channels*2)
        self.CPF_2 = AGS(in_channels*4)
        
        self.FA_16 = FeatureAtt(in_channels*4, 192)
        self.FA_8 = FeatureAtt(in_channels*2, 64)
        
        self.GSAM_8_d = ECF(in_channels*2, 64)
        self.GSAM_16_d = ECF(in_channels*4, 192)
        self.GSAM_32_d = ECF(in_channels*6, 160)
        
    def forward(self, x, imgs):
        conv1 = self.conv1(x)  
        conv1 = self.GSAM_8_d(conv1, imgs[1])
        
        conv2 = self.conv2(conv1)  
        conv2 = self.GSAM_16_d(conv2, imgs[2])
        
        conv3 = self.conv3(conv2)  
        conv3 = self.GSAM_32_d(conv3, imgs[3])
        
        conv3_up = self.conv3_up(conv3)  
        conv2 = self.CPF_2(self.FA_16(conv3_up, imgs[2]), self.FA_16(conv2, imgs[2]))  
        conv2 = self.agg_0(conv2) 

        conv2_up = self.conv2_up(conv2)  
        conv1 =  self.CPF_1(self.FA_8(conv2_up, imgs[1]), self.FA_8(conv1, imgs[1])) 
        conv1 = self.agg_1(conv1)  
        
        conv = self.conv1_up(conv1)  

        return conv  

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.max_pool = nn.AdaptiveMaxPool2d(1)  

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  
        self.relu1 = nn.ReLU()  
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  
        out = avg_out + max_out  
        return self.sigmoid(out)  
    
class SpatialAttention(nn.Module): 
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

class ECF(nn.Module):
    def __init__(self, in_channels, feat_channels):  
        super(ECF, self).__init__()   
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = BasicConv(in_channels, in_channels, is_3d=True, bn=False, relu=False, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv2 = BasicConv(in_channels, in_channels, is_3d=True, bn=True, relu=False, kernel_size=(1, 3, 3), padding=(0, 1, 1)) 
        self.conv3 = BasicConv(in_channels, in_channels, is_3d=True, bn=False, relu=False, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv4 = BasicConv(in_channels, in_channels, is_3d=True, bn=True, relu=False, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv5 = BasicConv(in_channels, in_channels, is_3d=True, bn=False, relu=False, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv6 = BasicConv(in_channels, in_channels, is_3d=True, bn=True, relu=False, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.semantic = nn.Sequential(  
                        BasicConv(feat_channels, feat_channels // 2, kernel_size=1, stride=1, padding=0), 
                        nn.Conv2d(feat_channels // 2, in_channels, 1))
        self.Spatial_att = SpatialAttention(kernel_size=7)
        self.Channel_att = ChannelAttention(in_channels)
        
    def forward(self, x, feat):    
        feat = self.semantic(feat)   
        att = self.Spatial_att(feat).unsqueeze(2)
        att_1 = self.Channel_att(feat).unsqueeze(2)
        rem = x
        x = x + feat.unsqueeze(2)
        x = self.conv2(self.conv1(x)) 
        x = self.relu(x + rem)        
        x_ = att * x
        x_1 = att_1 * x
        x = x + self.conv4(self.conv3(x_)) + self.conv6(self.conv5(x_1))
        return x

class Context_Stereo(nn.Module):
    def __init__(self, maxdisp):
        super(Context_Stereo, self).__init__()
        self.maxdisp = maxdisp 
        self.feature = Feature()
        self.feature_up = FeatUp()

        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU()
            )
        
        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x(32, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
            )

        self.conv = BasicConv(96, 48, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(48, 48, kernel_size=1, padding=0, stride=1)
        self.hourglass_fusion = hourglass_fusion(8)
        self.corr_stem = BasicConv(1, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        
    def forward(self, left, right): 
        features_left = self.feature(left)   
        features_right = self.feature(right)
        features_left, features_right = self.feature_up(features_left, features_right) 
        stem_2x = self.stem_2(left)   
        stem_4x = self.stem_4(stem_2x)  
        stem_2y = self.stem_2(right)  
        stem_4y = self.stem_4(stem_2y)  
        
        features_left[0] = torch.cat((features_left[0], stem_4x), 1)  
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)  

        match_left = self.desc(self.conv(features_left[0]))  
        match_right = self.desc(self.conv(features_right[0]))  

        corr_volume = build_norm_correlation_volume(match_left, match_right, self.maxdisp//4)  
        corr_volume = self.corr_stem(corr_volume)   
        cost = self.hourglass_fusion(corr_volume, features_left)  

        xspx = self.spx_4(features_left[0])  
        xspx = self.spx_2(xspx, stem_2x)  
        spx_pred = self.spx(xspx)  
        spx_pred = F.softmax(spx_pred, 1)  

        disp_samples = torch.arange(0, self.maxdisp//4, dtype=cost.dtype, device=cost.device)
        disp_samples = disp_samples.view(1, self.maxdisp//4, 1, 1).repeat(cost.shape[0],1,cost.shape[3],cost.shape[4])
        pred = regression_topk(cost.squeeze(1), disp_samples, 2)  
        pred_up = context_upsample(pred, spx_pred) 


        if self.training:
            return [pred_up*4, pred.squeeze(1)*4]  

        else:
            return [pred_up*4]  
