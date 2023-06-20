import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from .blocks import Conv2dBlock, SPADEResnetBlock, GatedConv2d, TransposeGatedConv2d

##########################################
class fg_inpainting(nn.Module):
    def __init__(self, cfg):
        super(fg_inpainting, self).__init__()

        input_nc = cfg['input_nc']
        ngf = 64
        output_nc = cfg['output_nc']
        lab_nc = 16
        g_norm = cfg['G_norm_type']

        # self.n_downsample = 3

        self.init_conv = Conv2dBlock(input_nc + 1 + 1 + 3, ngf, kernel_size=7, stride=1, padding=3, norm=g_norm, activation='lrelu')

        self.enc1 = GatedConv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.enc2 = GatedConv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.enc22 = GatedConv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        self.enc3 = GatedConv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.enc32 = GatedConv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        self.enc4 = GatedConv2d(ngf * 8, ngf * 16, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.enc5 = GatedConv2d(ngf * 16, ngf * 16, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        self.enc6 = GatedConv2d(ngf * 16, ngf * 16, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        
        # self.dec1 = TransposeGatedConv2d(ngf * 16 + ngf * 16, ngf * 16, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        self.dec2 = TransposeGatedConv2d(ngf * 16 + ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        self.dec22 = GatedConv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        self.dec3 = TransposeGatedConv2d(ngf * 8 + ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        self.dec32 = GatedConv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        self.dec4 = TransposeGatedConv2d(ngf * 4 + ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        self.dec43 = GatedConv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        self.dec5 = TransposeGatedConv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        self.dec6 = GatedConv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')

        # out
        self.conv_img = Conv2dBlock(ngf, output_nc, kernel_size=3, stride=1, padding=1, norm='none', activation='tanh')


    def forward(self, input, inst_mask, mask, class_map):
        x_in = self.init_conv(torch.cat((input, inst_mask, mask, class_map), dim=1))
        e1 = self.enc1(x_in)
        e2 = self.enc2(e1)
        e2 = self.enc22(e2)
        e3 = self.enc3(e2)
        e3 = self.enc32(e3)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        
        # d1 = self.dec1(e6, skip=e4)
        d2 = self.dec2(e6, skip=e3)
        d2 = self.dec22(d2)
        d3 = self.dec3(d2, skip=e2)
        d3 = self.dec32(d3)
        d4 = self.dec4(d3, skip=e1)
        d4 = self.dec43(d4)
        d5 = self.dec5(d4)
        d6 = self.dec6(d5)

        out = self.conv_img(d6)

        return out