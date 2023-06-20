import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from .blocks import Conv2dBlock, GatedConv2d, TransposeGatedConv2d, UpConv2dBlock

##########################################
class F_stage(nn.Module):
    def __init__(self, cfg):
        super(F_stage, self).__init__()

        input_nc = cfg['input_nc']
        ngf = 32
        output_nc = cfg['output_nc']
        g_norm = cfg['G_norm_type']
        lab_nc = cfg['lab_dim']

        self.n_downsample = 3
        
        ######### fusion
        self.init_conv = Conv2dBlock(input_nc + 1 + 1, ngf, kernel_size=7, stride=1, padding=3, norm=g_norm, activation='lrelu')

        self.enc1 = GatedConv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.enc2 = GatedConv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        self.enc3 = GatedConv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.enc4 = GatedConv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        self.enc5 = GatedConv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.enc6 = GatedConv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')

        # Upsampling
        self.dec2 = TransposeGatedConv2d(ngf * 8 + ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        self.dec3 = TransposeGatedConv2d(ngf * 4 + ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        self.dec4 = TransposeGatedConv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        self.dec5 = GatedConv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')

        # out
        self.conv_img = Conv2dBlock(ngf, output_nc, kernel_size=3, stride=1, padding=1, norm='none', activation='tanh')


    def forward(self, masked_G, segmap, mask, edge_map):
        ######### fusion
        f_input = torch.cat((masked_G, mask, edge_map), dim=1)
        x_in = self.init_conv(f_input)
        e1 = self.enc1(x_in)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)

        d2 = self.dec2(e6, skip=e4)
        d3 = self.dec3(d2, skip=e2)
        d4 = self.dec4(d3)
        d5 = self.dec5(d4)

        out = self.conv_img(d5)

        return out * mask + masked_G