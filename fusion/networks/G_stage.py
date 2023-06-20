import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .inst_Gs import inst_Gs
from .bg_G import bg_G
import copy

##########################################
class G_stage(nn.Module):
    def __init__(self, cfg):
        super(G_stage, self).__init__()
        
        # fg inpainiting
        self.inst_Gs = inst_Gs(cfg)
        
        state_dict = torch.load('submodels/object_inpainting.pth')['netG']
        state_dict_v2 = copy.deepcopy(state_dict)
        for key in state_dict:
            if 'fg_inpainting.' in key:
                state_dict_v2[key.replace('fg_inpainting.', '')] = state_dict_v2.pop(key)
        self.inst_Gs.netG_fg_inpainting.load_state_dict(state_dict_v2)
        print('load inst model: inpainting')
        
        # style
        state_dict = torch.load('submodels/object_style.pth')['netG']
        state_dict_v2 = copy.deepcopy(state_dict)
        for key in state_dict:
            if 'fg_style.' in key:
                state_dict_v2[key.replace('fg_style.', '')] = state_dict_v2.pop(key)
        self.inst_Gs.netG_fg_style.load_state_dict(state_dict_v2)
        print('load inst model: style')
        
        # bg
        self.bg_G = bg_G(cfg)
        state_dict = torch.load('submodels/bg.pth')['netG_BG']
        self.bg_G.load_state_dict(state_dict)
        print('load inst model: bg')
        

    def forward(self, gt, masked_img, mask, fg_mask, 
                    car_gt_crop, car_gt_crop2048, car_masked_img_crop, car_ints_mask_crop, car_mask_crop, car_inst_cors, 
                    person_gt_crop, person_gt_crop2048, person_masked_img_crop, person_ints_mask_crop, person_mask_crop, person_inst_cors, 
                    segmap_edge, segmap, bg_mask, 
                    car_sty_images, car_sty_inst_images, person_sty_images, person_sty_inst_images):
                    
        inst_output_list, fg_output = self.inst_Gs(gt, masked_img, mask, fg_mask, 
                                        car_gt_crop, car_gt_crop2048, car_masked_img_crop, car_ints_mask_crop, car_mask_crop, car_inst_cors, 
                                        person_gt_crop, person_gt_crop2048, person_masked_img_crop, person_ints_mask_crop, person_mask_crop, person_inst_cors,
                                        car_sty_images, car_sty_inst_images, person_sty_images, person_sty_inst_images)
        
        bg_output = self.bg_G(torch.cat((masked_img, segmap_edge*bg_mask, mask), dim=1), segmap, mask, bg_mask)

        return inst_output_list, fg_output, bg_output