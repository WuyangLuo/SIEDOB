import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from .blocks import Conv2dBlock, SPADE, GatedConv2d, TransposeGatedConv2d

##########################################
class bg_G(nn.Module):
    def __init__(self, cfg):
        super(bg_G, self).__init__()

        input_nc = cfg['input_nc']
        ngf = 32
        output_nc = cfg['output_nc']
        lab_nc = cfg['lab_dim']
        g_norm = cfg['G_norm_type']
        cond_nc = lab_nc + 1 + 1

        self.n_downsample = 3

        self.init_conv = Conv2dBlock(input_nc + lab_nc + 1 + 1, ngf, kernel_size=7, stride=1, padding=3, norm=g_norm, activation='lrelu')

        self.enc1 = GatedConv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.enc2 = GatedConv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        self.enc3 = GatedConv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.enc4 = GatedConv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        self.enc5 = GatedConv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.enc6 = GatedConv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        self.enc7 = GatedConv2d(ngf * 8, ngf * 8, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.enc8 = GatedConv2d(ngf * 8, ngf * 16, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')

        # Upsampling
        self.dec1 = TransposeGatedConv2d(ngf * 16 + ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        self.style1 = StyleProPagate(ngf * 8, lab_nc)
        self.dec2 = TransposeGatedConv2d(ngf * 8 + ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        self.style2 = StyleProPagate(ngf * 4, lab_nc)
        self.dec3 = TransposeGatedConv2d(ngf * 4 + ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        self.style3 = StyleProPagate(ngf * 2, lab_nc)
        self.dec4 = TransposeGatedConv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        # self.style4 = StyleProPagate(ngf * 2, lab_nc)
        self.dec5 = GatedConv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')

        # out
        self.conv_img = Conv2dBlock(ngf, output_nc, kernel_size=3, stride=1, padding=1, norm='none', activation='tanh')


    def forward(self, input, segmap, mask, bg_mask):
        x_in = self.init_conv(input)
        e1 = self.enc1(x_in)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        d1 = self.dec1(e8, skip=e6)
        d1 = self.style1(d1, segmap, bg_mask, mask)
        d2 = self.dec2(d1, skip=e4)
        d2 = self.style2(d2, segmap, bg_mask, mask)
        d3 = self.dec3(d2, skip=e2)
        d3 = self.style3(d3, segmap, bg_mask, mask)
        d4 = self.dec4(d3)
        # d4 = self.style4(d4, segmap, bg_mask, mask)
        d5 = self.dec5(d4)

        out = self.conv_img(F.leaky_relu(d5, 2e-1))

        return out



class StyleProPagate(nn.Module):
    def __init__(self, dim_in, label_nc):
        super(StyleProPagate, self).__init__()
        
        s_dim = 64

        self.conv = Conv2dBlock(dim_in, s_dim, kernel_size=3, stride=1, padding=1, norm='none', activation='lrelu')
        
        self.stylelayer = StyleLayer(dim_in, s_dim+label_nc)

    def forward(self, featmap_in, segmap, bg_mask, mask, class_selected=[24,25,26,27,28,29,30,31,32,33]):
        featmap = self.conv(featmap_in)
        
        segmap = F.interpolate(segmap, size=featmap.size()[2:], mode='nearest')
        bg_mask = F.interpolate(bg_mask, size=featmap.size()[2:], mode='nearest')
        mask = F.interpolate(mask, size=featmap.size()[2:], mode='nearest')
        
        ctx_segmap = segmap * bg_mask * (1. - mask)
        bg_segmap = segmap * bg_mask

        b_size = featmap.shape[0]
        f_size = featmap.shape[1]
        s_size = segmap.shape[1]

        codes_vector = torch.zeros((b_size, s_size, f_size), dtype=featmap.dtype, device=featmap.device)

        for i in range(b_size):
            for j in range(s_size):
                if j not in class_selected:  # 不计算前景（class_selected）
                    component_mask_area = torch.sum(ctx_segmap.bool()[i, j])

                    if component_mask_area > 0:
                        codes_component_feature = featmap[i].masked_select(ctx_segmap.bool()[i, j]).reshape(f_size, component_mask_area).mean(1)
                        codes_vector[i][j] = codes_component_feature

        bs, N, h, w = bg_segmap.size()
        _, N, D = codes_vector.size()
        style_codesT = codes_vector.permute(0, 2, 1)  # B x D x N
        bg_segmap1 = bg_segmap.view(bs, N, -1)  # B x N x HW
        style_map = torch.matmul(style_codesT, bg_segmap1)
        style_map = style_map.view(bs, D, h, w)  # B x D x H x W
        
        # print(style_map.size())
        # print(bg_segmap.size())
        
        out_map = torch.cat((style_map, bg_segmap), dim=1)
        
        out = self.stylelayer(featmap_in, out_map)

        return out

class StyleLayer(nn.Module):
    def __init__(self, dim, label_nc):
        super(StyleLayer, self).__init__()

        # create conv layers
        self.conv_0 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.spade = SPADE(dim, label_nc)

    def forward(self, x, seg):
        dx = self.conv_0(x)
        dx = self.spade(dx, seg)
        dx = self.actvn(dx)
        return dx

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)