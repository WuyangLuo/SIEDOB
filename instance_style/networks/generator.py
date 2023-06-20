import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from .fg_style import fg_style

##########################################
class generator(nn.Module):
    def __init__(self, cfg):
        super(generator, self).__init__()

        self.fg_style = fg_style(cfg)


    def forward(self, cls_map, sty_img, sty_cls_map, inst_mask):
        style_img, sty_code, cycle_sc = self.fg_style(cls_map, sty_img, sty_cls_map, inst_mask)

        return style_img, sty_code, cycle_sc