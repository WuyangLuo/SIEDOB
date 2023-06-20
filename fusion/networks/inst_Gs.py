import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.fg_inpainting import fg_inpainting
from networks.fg_style import fg_style
from .blocks import Conv2dBlock, SPADEResnetBlock

##########################################
class inst_Gs(nn.Module):
    def __init__(self, cfg):
        super(inst_Gs, self).__init__()
        
        self.cfg = cfg
        self.max_num = self.cfg['max_num']
        self.inst_size = self.cfg['inst_size']

        self.netG_fg_inpainting = fg_inpainting(cfg)
        self.netG_fg_style = fg_style(cfg)

    def forward(self, gt, masked_img, mask, fg_mask, 
                    car_gt_crop, car_gt_crop2048, car_masked_img_crop, car_ints_mask_crop, car_mask_crop, car_inst_cors, 
                    person_gt_crop, person_gt_crop2048, person_masked_img_crop, person_ints_mask_crop, person_mask_crop, person_inst_cors,
                    car_sty_images, car_sty_inst_images, person_sty_images, person_sty_inst_images
                    ):

        ### car ###
        car_input = car_masked_img_crop * car_ints_mask_crop
        car_inst_mask_filtering, car_input_filtering, car_gt_filtering, car_mask_filtering, car_is_none, car_num \
            = self.filtering_fg_mask_batch(car_ints_mask_crop, car_input, car_gt_crop2048, car_mask_crop)
        if car_num > 0:
            cls = torch.tensor([1, 0, 0]).unsqueeze(0).cuda()
            car_output_list = []
            for idx in range(car_num):
                curr_car_inst_mask_filtering = car_inst_mask_filtering[idx:idx + 1]
                curr_car_input_filtering = car_input_filtering[idx:idx + 1]
                curr_car_gt_filtering = car_gt_filtering[idx:idx + 1]
                curr_car_mask_filtering = car_mask_filtering[idx:idx + 1]
                class_map = (curr_car_inst_mask_filtering * cls.unsqueeze(-1).unsqueeze(-1))

                curr_car_mask_filtering = curr_car_mask_filtering * curr_car_inst_mask_filtering
                # style generation
                if torch.sum(curr_car_mask_filtering) == torch.sum(curr_car_inst_mask_filtering):
                    NUM_STY = car_sty_inst_images.size()[0]
                    iio = idx % NUM_STY
                    sty_img = car_sty_images[iio:iio+1]
                    inst = car_sty_inst_images[iio:iio + 1]
                    sty_cls_map = (inst * cls.unsqueeze(-1).unsqueeze(-1))
                    out, _, _ = self.netG_fg_style(class_map, sty_img*inst, sty_cls_map, curr_car_inst_mask_filtering)
                # inpainting
                else:
                    out = self.netG_fg_inpainting(curr_car_input_filtering, curr_car_inst_mask_filtering, curr_car_mask_filtering, class_map)

                car_output_list.append(out)
                car_output = torch.cat((car_output_list), dim=0)

            car_output_filling = self.filling_fg_mask_batch(car_output, car_is_none, car_input.size())
        # if there is no car, 
        else:
            car_output = torch.zeros((1,3,self.inst_size,self.inst_size)).cuda()
            car_output_filling = torch.zeros(gt.size()).cuda()

        ### person ###
        person_input = person_masked_img_crop * person_ints_mask_crop
        person_inst_mask_filtering, person_input_filtering, person_gt_filtering, person_mask_filtering, person_is_none, person_num \
            = self.filtering_fg_mask_batch(person_ints_mask_crop, person_input, person_gt_crop2048, person_mask_crop)
        if person_num > 0:
            cls = torch.tensor([0,1,0]).unsqueeze(0).cuda()
            person_output_list = []
            for idx in range(person_num):
                curr_person_inst_mask_filtering = person_inst_mask_filtering[idx:idx + 1]
                curr_person_input_filtering = person_input_filtering[idx:idx + 1]
                curr_person_gt_filtering = person_gt_filtering[idx:idx + 1]
                curr_person_mask_filtering = person_mask_filtering[idx:idx + 1]
                class_map = (curr_person_inst_mask_filtering * cls.unsqueeze(-1).unsqueeze(-1))

                curr_person_mask_filtering = curr_person_mask_filtering * curr_person_inst_mask_filtering
                # style generation
                if torch.sum(curr_person_mask_filtering) == torch.sum(curr_person_inst_mask_filtering):
                    NUM_STY = car_sty_inst_images.size()[0]
                    iio = idx % NUM_STY
                    sty_img = person_sty_images[iio:iio+1]
                    inst = person_sty_inst_images[iio:iio + 1]
                    sty_cls_map = (inst * cls.unsqueeze(-1).unsqueeze(-1))
                    out, _, _ = self.netG_fg_style(class_map, sty_img*inst, sty_cls_map, curr_person_inst_mask_filtering)
                # inpainting
                else:
                    out = self.netG_fg_inpainting(curr_person_input_filtering, curr_person_inst_mask_filtering, curr_person_mask_filtering, class_map)

                person_output_list.append(out)
                person_output = torch.cat((person_output_list), dim=0)

            person_output_filling = self.filling_fg_mask_batch(person_output, person_is_none, person_input.size())
        else:
            person_output = torch.zeros((1,3,self.inst_size,self.inst_size)).cuda()
            person_output_filling = torch.zeros(gt.size()).cuda()
        
        embed_car_output, mask_carcaracrcar = self.embed_crop(car_output_filling, car_ints_mask_crop, car_inst_cors, batch_size=gt.size()[0], max_inst_num=self.max_num)
        embed_person_output, _ = self.embed_crop(person_output_filling, person_ints_mask_crop, person_inst_cors, batch_size=gt.size()[0], max_inst_num=self.max_num)
        embed_output = embed_car_output + embed_person_output
        
        fg_gen_masked_img = (embed_output * fg_mask * mask) + masked_img * (1. - fg_mask * mask)  # detach()
        gt_fg_gen = (embed_output * fg_mask * mask) + gt * (1. - fg_mask * mask)

        return [car_input_filtering, car_output, car_inst_mask_filtering, car_mask_filtering, car_num, \
               person_input_filtering, person_output, person_inst_mask_filtering, person_mask_filtering, person_num, \
               embed_car_output, embed_person_output, embed_output, gt_fg_gen], fg_gen_masked_img.detach()

    # embed a category of objects back to their original position
    def embed_crop(self, output, ints_mask, cors, batch_size, max_inst_num):
        N, c, _, _ = output.size()
        embedded_images = torch.zeros((batch_size, c, 256, 256)).cuda()
        img = torch.zeros((1, 1, 256, 256)).cuda()
        
        b = 0
        for n in range(N):
            if n > 0 and n % max_inst_num == 0:
                b += 1
            if torch.sum(cors[n]) == 0:
                pass
            else:
                x_min, x_max, y_min, y_max = cors[n]
                curr_output = F.interpolate(output[n:n+1,:,:,:], (x_max-x_min, y_max-y_min), mode='bilinear')
                curr_mask = F.interpolate(ints_mask[n:n+1,:,:,:], (x_max-x_min, y_max-y_min), mode='nearest')
                img = torch.zeros((1, c, 256, 256)).cuda()
                mask = torch.zeros((1, 1, 256, 256)).cuda()
                img[:, :, x_min:x_max, y_min:y_max] = curr_output
                mask[:, :, x_min:x_max, y_min:y_max] = curr_mask
                
                embedded_images[b:b+1] = embedded_images[b:b+1] * (1. - mask) + img * mask
        return embedded_images, img

    # only generate non-empty object
    def filtering_fg_mask_batch(self, ints_mask_crop, fg_input, gt_crop, mask_crop):
        inst_mask_filtering = []
        input_filtering = []
        gt_filtering = []
        mask_filtering = []
        is_none = []
        num = 0
        for n in range(ints_mask_crop.size()[0]):  # num
            if torch.sum(ints_mask_crop[n]) > 0:
                inst_mask_filtering.append(ints_mask_crop[n:n+1])
                input_filtering.append(fg_input[n:n+1])
                gt_filtering.append(gt_crop[n:n+1])
                mask_filtering.append(mask_crop[n:n+1])
                is_none.append(False)
                num += 1
            else:
                is_none.append(True)                
                
        if num > 0:
            inst_mask_filtering = torch.cat(inst_mask_filtering, dim=0)
            input_filtering = torch.cat(input_filtering, dim=0)
            gt_filtering = torch.cat(gt_filtering, dim=0)
            mask_filtering = torch.cat(mask_filtering, dim=0)
        else:
            size = self.inst_size
            inst_mask_filtering = torch.zeros((1,3,size,size)).cuda()
            input_filtering = torch.zeros((1,3,size,size)).cuda()
            gt_filtering = torch.zeros((1,3,size,size)).cuda()
            mask_filtering = torch.zeros((1,3,size,size)).cuda()
        
        return inst_mask_filtering, input_filtering, gt_filtering, mask_filtering, is_none, num
    
    # re-construct a object batch
    def filling_fg_mask_batch(self, output, is_none, size):
        output_filling = []
        n = 0
        for curr_none in is_none:  # batch size
            if curr_none:
                output_filling.append(torch.zeros((1, size[1], size[2], size[3])).cuda())
            else:
                output_filling.append(output[n:n+1])
                n += 1
        output_filling = torch.cat(output_filling, dim=0)
        return output_filling