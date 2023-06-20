import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from .blocks import Conv2dBlock, UpConv2dBlock

class SNPatchDiscriminator(nn.Module):
    def __init__(self, cfg, input_nc=None, num_layer=6):
        super(SNPatchDiscriminator, self).__init__()
        
        if input_nc == None:
            input_nc = cfg['input_nc'] + 3
        ndf = cfg['ndf']
        self.num_layer = num_layer


        dis_layers = [Conv2dBlock(input_nc, ndf, kernel_size=5, stride=2, padding=2, norm='sn', activation='lrelu')]
        for i in range(self.num_layer-2):
            mult = 2 ** i
            if mult < 4:
                dis_layers += [Conv2dBlock(ndf * mult, ndf * mult * 2, kernel_size=5, stride=2, padding=2, norm='sn', activation='lrelu')]
            else:
                mult = 4
                dis_layers += [Conv2dBlock(ndf * mult, ndf * mult, kernel_size=5, stride=2, padding=2, norm='sn', activation='lrelu')]
        dis_layers += [Conv2dBlock(ndf * 4, 3, kernel_size=5, stride=2, padding=2, norm='sn', activation='lrelu')]

        self.dis_model = nn.Sequential(*dis_layers)

    def forward(self, input):

        d = self.dis_model(input)  # torch.Size([9, 3, 4, 4])

        return d