import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from .fg_inpainting import fg_inpainting

##########################################
class generator(nn.Module):
    def __init__(self, cfg):
        super(generator, self).__init__()

        self.fg_inpainting = fg_inpainting(cfg)


    def forward(self, input, inst_mask, mask, class_map):
        inp_img = self.fg_inpainting(input, inst_mask, mask, class_map)
        
        masked_inp = inp_img * mask + input * (1. - mask)

        return inp_img, masked_inp