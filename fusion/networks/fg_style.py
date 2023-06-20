import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from .blocks import Conv2dBlock, LinearBlock, styleSPADEResnetBlock, ResnetBlock
##########################################

class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ='lrelu', pad_type='reflect'):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        # self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##########################################
class fg_style(nn.Module):
    def __init__(self, cfg):
        super(fg_style, self).__init__()

        input_nc = cfg['input_nc']
        ngf = 32
        output_nc = cfg['output_nc']
        sty_nc = 128
        ins_nc = 3
        g_norm = 'in'

        self.enc_style = StyleEncoder(4, input_nc + 3, 32, sty_nc, norm='none')

        self.init_conv = Conv2dBlock(3, ngf, kernel_size=7, stride=1, padding=3, norm=g_norm, activation='lrelu')

        self.enc1 = Conv2dBlock(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.enc2 = Conv2dBlock(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.enc3 = Conv2dBlock(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')

        self.resblk1 = ResnetBlock(ngf * 8)
        self.resblk2 = ResnetBlock(ngf * 8)
        self.resblk3 = ResnetBlock(ngf * 8)
        self.resblk4 = ResnetBlock(ngf * 8)

        self.spade1_1 = styleSPADEResnetBlock(ngf * 8, ngf * 8, ins_nc, sty_nc)
        self.spade1_2 = styleSPADEResnetBlock(ngf * 8, ngf * 8, ins_nc, sty_nc)
        self.conv1 = ResnetBlock(ngf * 8, ngf * 4)

        self.spade2_1 = styleSPADEResnetBlock(ngf * 4, ngf * 4, ins_nc, sty_nc)
        self.spade2_2 = styleSPADEResnetBlock(ngf * 4, ngf * 4, ins_nc, sty_nc)
        self.conv2 = ResnetBlock(ngf * 4, ngf * 2)

        self.spade3 = styleSPADEResnetBlock(ngf * 2, ngf * 2, ins_nc, sty_nc)
        self.conv3 = ResnetBlock(ngf * 2, ngf * 1)

        # out
        self.conv_img = Conv2dBlock(ngf, output_nc, kernel_size=3, stride=1, padding=1, norm='none', activation='tanh')

        self.up = nn.Upsample(scale_factor=2)

        self.cycle_style = StyleEncoder(4, input_nc + 3, 32, sty_nc, norm='none')

    def forward(self, cls_map, sty_img, sty_cls_map, inst_mask):

        # print('cls_map: ',cls_map.size())
        # print('sty_img: ',sty_img.size())
        # print('sty_cls_map: ',sty_cls_map.size())
        # print('inst_mask: ',inst_mask.size())

        # cls_map:  torch.Size([1, 3, 64, 64])
        # sty_img:  torch.Size([0, 3, 64, 64])
        # sty_cls_map:  torch.Size([0, 3, 64, 64])
        # inst_mask:  torch.Size([1, 1, 64, 64])
        # inst_mask:  torch.Size([1, 1, 64, 64])
        # cls_map:  torch.Size([1, 3, 64, 64])
        # x:  torch.Size([0, 3, 64, 64])

        x_in = cls_map
        sty_in = torch.cat((sty_img, sty_cls_map), dim=1)

        sty_code = self.enc_style(sty_in)

        x = self.init_conv(x_in)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        x = self.resblk1(x)
        x = self.resblk2(x)
        x = self.resblk3(x)
        x = self.resblk4(x)

        x = self.spade1_1(x, cls_map, sty_code)
        x = self.spade1_2(x, cls_map, sty_code)
        x = self.conv1(x)
        x = self.up(x)

        x = self.spade2_1(x, cls_map, sty_code)
        x = self.spade2_2(x, cls_map, sty_code)
        x = self.conv2(x)
        x = self.up(x)

        x = self.spade3(x, cls_map, sty_code)
        x = self.conv3(x)
        x = self.up(x)

        x = self.conv_img(F.leaky_relu(x, 2e-1))

        # print('inst_mask: ',inst_mask.size())
        # print('cls_map: ',cls_map.size())
        # print('x: ',x.size())

        cycle_sc = self.cycle_style(torch.cat((x*inst_mask, cls_map), dim=1))

        return x, sty_code, cycle_sc