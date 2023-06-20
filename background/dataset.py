import os
import cv2
import math
import numpy as np
from torch.utils.data import Dataset
import os.path
import random
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from utils import make_dataset
from PIL import Image, ImageDraw


class SIE_Dataset(Dataset):
    def __init__(self, cfg, dataset_root, split='train', mask_type='ff_mask', dataset_name=''):
        self.split = split
        self.cfg = cfg
        self.mask_type = mask_type
        self.dataset_name = dataset_name
        self.lab_nc = self.cfg['lab_dim']

        self.dir_img = os.path.join(dataset_root, self.split, 'images')
        self.dir_lab = os.path.join(dataset_root, self.split, 'labels')
        self.dir_ins = os.path.join(dataset_root, self.split, 'inst_map')
        self.dir_img2048 = os.path.join(dataset_root, self.split, 'images2048')
        name_list = os.listdir(self.dir_img)
        self.name_list = [n[:-4] for n in name_list if n.endswith('jpg')]

        if self.mask_type == 'ff_mask':
            self.mask_path = ''  # path to pre-generated free-form mask
            mask_list = os.listdir(self.mask_path)
            mask_list.sort()
            self.mask_list = mask_list[:len(self.name_list)]
        else:
            pass
        
        if self.split == 'test':
            self.name_list.sort()
        
        self.max_num = self.cfg['max_num']
        self.inst_size = self.cfg['inst_size']

    def __getitem__(self, index):
        name = self.name_list[index]
        # input data
        img = cv2.imread(os.path.join(self.dir_img, name + '.jpg'))
        lab = cv2.imread(os.path.join(self.dir_lab, name + '.png'), 0)
        # Change the 'unknown' label to the last label
        lab[lab == 0] = self.lab_nc - 1

        if self.dataset_name == 'cityscapes512x256':
            inst_map = Image.open(os.path.join(self.dir_ins, name + '.png'))
            inst_map = np.array(inst_map, dtype=np.int32)

        if self.dataset_name == 'cityscapes512x256':
            fg_class_list = [24, 26]  # car and person
            bg_class_list = [6, 7, 8 ,9, 10, 11, 12, 21, 22, 23]
        
        if self.split == 'train':
            gt2048 = cv2.imread(os.path.join(self.dir_img2048, name + '.jpg'))
            # crop
            h, w, _ = img.shape
            w_l = 0
            h_l = 0
            if w > 256:
                w_l = random.randint(0, w - 256)
            if h > 256:
                h_l = random.randint(0, h - 256)
            
            img = img[h_l:h_l+256, w_l:w_l+256]
            lab = lab[h_l:h_l+256, w_l:w_l+256]
            h_l_2048 = min(2048, h_l*4)
            w_l_2048 = min(2048, w_l*4)
            gt2048 = gt2048[h_l_2048:h_l_2048+1024, w_l_2048:w_l_2048+1024]
            if self.dataset_name == 'cityscapes512x256':
                inst_map = inst_map[h_l:h_l+256, w_l:w_l+256]
            # flip
            if random.random() > 0.5:
                # print('filp')
                img = np.flip(img,axis=1).copy()
                lab = np.flip(lab,axis=1).copy()
                gt2048 = np.flip(gt2048,axis=1).copy()
                if self.dataset_name == 'cityscapes512x256':
                    inst_map = np.flip(inst_map,axis=1).copy()
        
        else:
            gt2048 = np.zeros((1024, 1024, 3))
        
        # id defination
        # cityscapes:
        #   fg: person:24, car:26,
        #   bg: road:7, sidewalks:8, vegetation:21, sky:23
        
        # select inst id
        if self.dataset_name == 'cityscapes512x256':
            inst_ids = np.unique(inst_map)
            inst_ids = inst_ids.tolist()
            inst_ids = [i for i in inst_ids if i>=1000]  # filter out non-instance masks

        
        
        lab_ori = lab.copy()
        
        # extract all ids of the current input "lab"
        lab_ids = np.unique(lab)
        lab_ids = lab_ids.tolist()
        selected_lab_id = random.choice(lab_ids)
        
        img = get_transform(img)
        gt2048 = get_transform(gt2048)
        lab = get_transform(lab, normalize=False)
        lab = lab * 255.0

        fg_inst_mask_list = []
        for iid in inst_ids:
            curr_iid = np.array(np.equal(inst_map, iid).astype(np.float32))
            fg_inst_mask_list.append(curr_iid)
        
        mask_type = 0
        if self.split == 'train':
            mask_type = index % 3
            # ff
            if mask_type == 0:
                mask = brush_stroke_mask()
                mask = mask.reshape((1,) + mask.shape).astype(np.float32)
            # right
            elif mask_type == 1:
                mask = self.load_right_mask(self.cfg['crop_size'])
            # center
            elif mask_type == 2:
                mask = self.load_center_mask(self.cfg['crop_size'], split='train')

            if random.random() > 0.5:
                # class
                class_mask = self.load_inst_mask(self.cfg['crop_size'], lab_ori, fg_class_list, bg_class_list, fg_inst_mask_list)
                mask = self.mask_binary(mask + class_mask)
            
            mask = self.mask_binary(mask)
        
        else:
            mask = cv2.imread(os.path.join(self.mask_path, self.mask_list[index]), 0)
            mask = mask.reshape((1,) + mask.shape).astype(np.float32)

        # generate inp_mask, gen_mask
        if self.dataset_name == 'cityscapes512x256':
            lab_masked = lab_ori * mask[0]
            lab_masked_inv = lab_ori * (1. - mask[0])
            inst_masked = inst_map * mask[0]
            inst_masked_inv = inst_map * (1. - mask[0])
            
            fg_inp_inst = []
            fg_gen_inst = []
            bg_inp_inst = []
            bg_gen_inst = []
            fg_inp_classes = []
            fg_gen_classes = []
            
            
            # foreground object class
            for iid in inst_ids:
                curr_iid = np.array(np.equal(inst_masked, iid).astype(np.float32))
                if np.sum(curr_iid) > 0:  # iid in masked region
                    curr_class_id = lab_masked * curr_iid
                    curr_inst_label_ids = np.unique(curr_class_id)
                    curr_inst_label_ids = curr_inst_label_ids.tolist()
                    curr_inst_label_ids = [i for i in curr_inst_label_ids if i>0]
                    # instmap and segmap may not overlap exactly
                    num_max = 0.
                    max_class = -1.
                    for n_iid in curr_inst_label_ids:  # finding the semantic lab of the current instmap
                        if n_iid in fg_class_list:  # only for pre-definded fg class
                            curr_n_iid = np.array(np.equal(curr_class_id, n_iid).astype(np.float32))
                            if np.sum(curr_n_iid) > num_max:
                                num_max = np.sum(curr_n_iid)
                                max_class = n_iid
                    
                    curr_inst_mask = np.array(np.equal(inst_map, iid).astype(np.float32))
                    curr_inst_mask = np.array(np.equal(lab_ori*curr_inst_mask, max_class).astype(np.float32))
                    
                    if max_class in fg_class_list:
                        curr_iid_inv = np.array(np.equal(inst_masked_inv, iid).astype(np.float32))
                        if np.sum(curr_iid_inv) > 0:  # iid also appear in non-masked region
                            if mask_type != 1000000000000:
                                fg_inp_inst.append(curr_inst_mask)
                                fg_inp_classes.append(max_class)
                            else:
                                fg_gen_inst.append(curr_inst_mask)
                                fg_gen_classes.append(max_class)
                        else:
                            fg_gen_inst.append(curr_inst_mask)
                            fg_gen_classes.append(max_class)


            # background class
            for ibg in range(0, self.lab_nc+1): 
                if ibg in bg_class_list:
                    curr_ibg = np.array(np.equal(lab_ori, ibg).astype(np.float32)) * mask[0]
                    curr_ibg_inv = np.array(np.equal(lab_ori, ibg).astype(np.float32)) * (1. - mask[0])
                    if np.sum(curr_ibg) > 0 and np.sum(curr_ibg_inv) > 0:  # inpainting for current background
                        curr_mask = curr_ibg+curr_ibg_inv
                        if mask_type != 1000000000000:  # instmap and segmap may not overlap exactly
                            bg_inp_inst.append(curr_mask)
                        else:
                            bg_gen_inst.append(curr_ibg)
                    elif np.sum(curr_ibg) > 0 and np.sum(curr_ibg_inv) <= 0:
                        bg_gen_inst.append(curr_ibg)
            
            fg_inp_mask = np.zeros((256, 256), np.float32)
            for curr in fg_inp_inst:
                fg_inp_mask = fg_inp_mask + curr * (1. - fg_inp_mask)
            fg_inp_mask = self.mask_binary(fg_inp_mask)
            
            fg_gen_mask = np.zeros((256, 256), np.float32)
            for curr in fg_gen_inst:
                fg_gen_mask = fg_gen_mask + curr * (1. - fg_gen_mask)
            fg_gen_mask = self.mask_binary(fg_gen_mask)
            
            bg_inp_mask = np.zeros((256, 256), np.float32)
            for curr in bg_inp_inst:
                bg_inp_mask = bg_inp_mask + curr * (1. - bg_inp_mask)
            bg_inp_mask = self.mask_binary(bg_inp_mask)
            
            bg_gen_mask = np.zeros((256, 256), np.float32)
            for curr in bg_gen_inst:
                bg_gen_mask = bg_gen_mask + curr * (1. - bg_gen_mask)
            bg_gen_mask = self.mask_binary(bg_gen_mask)
            
            fg_mask_ori = self.mask_binary(fg_inp_mask + fg_gen_mask)
            bg_mask_ori = self.mask_binary(bg_inp_mask + bg_gen_mask)
            

            fg_inst = fg_inp_inst + fg_gen_inst
            fg_classes = fg_inp_classes + fg_gen_classes
            
            mask = torch.from_numpy(mask)
            masked_img = img * (1. - mask)

            # car
            car_gt_crop, car_gt_crop2048, car_masked_img_crop, car_lab_crop, car_ints_mask_crop, car_mask_crop, car_inst_cors, car_g_mask_inst \
                = self.inst_crop(img, gt2048, masked_img, lab, mask, fg_inst, fg_classes, max_num=self.max_num, 
                                 class_id=26, min_size=(0,0), crop_size=(64,64), sum_min_ratio=0.6, ratio_range=None)
            
            # person
            person_gt_crop, person_gt_crop2048, person_masked_img_crop, person_lab_crop, person_ints_mask_crop, person_mask_crop, person_inst_cors, person_g_mask_inst \
                = self.inst_crop(img, gt2048, masked_img, lab, mask, fg_inst, fg_classes, max_num=self.max_num, 
                                 class_id=24, min_size=(0,0), crop_size=(64,32), sum_min_ratio=0.35, ratio_range=(2,6))
            
            
            car_g_mask = np.zeros((256, 256), np.float32)
            for curr in car_g_mask_inst:
                car_g_mask = car_g_mask + curr * (1. - car_g_mask)
            car_g_mask = self.mask_binary(car_g_mask)
            
            person_g_mask = np.zeros((256, 256), np.float32)
            for curr in person_g_mask_inst:
                person_g_mask = person_g_mask + curr * (1. - person_g_mask)
            person_g_mask = self.mask_binary(person_g_mask)
            
            fg_mask = self.mask_binary(car_g_mask + person_g_mask)
            bg_mask = self.mask_binary(1. - fg_mask)

            car_g_mask = car_g_mask.reshape((1,) + car_g_mask.shape).astype(np.float32)
            car_g_mask = torch.from_numpy(car_g_mask)
            person_g_mask = person_g_mask.reshape((1,) + person_g_mask.shape).astype(np.float32)
            person_g_mask = torch.from_numpy(person_g_mask)

        fg_mask = fg_mask.reshape((1,) + fg_mask.shape).astype(np.float32)
        fg_mask = torch.from_numpy(fg_mask)
        bg_mask = bg_mask.reshape((1,) + bg_mask.shape).astype(np.float32)
        bg_mask = torch.from_numpy(bg_mask)
        bg_mask_ori = bg_mask_ori.reshape((1,) + bg_mask_ori.shape).astype(np.float32)
        bg_mask_ori = torch.from_numpy(bg_mask_ori)
        
        car_lab_masked_inst = car_lab_crop * car_ints_mask_crop
        car_colormap = inst2color(car_lab_masked_inst[0, 0].numpy(), size=(64,64))
        car_colormap = torch.from_numpy(car_colormap)

        person_lab_masked_inst = person_lab_crop * person_ints_mask_crop
        person_colormap = inst2color(person_lab_masked_inst[0, 0].numpy(), size=(64,32))
        person_colormap = torch.from_numpy(person_colormap)
        
        inst_map = inst_map.reshape((1,) + inst_map.shape).astype(np.float32)
        inst_map = torch.from_numpy(inst_map)
        
        class_num = 35
        bg_coo_map = np.zeros((class_num, 256, 256), np.float32)
        for i in range(len(bg_inp_inst)):
            bg_coo_map[i] = bg_inp_inst[i]

        return {'img': img, 'masked_img': masked_img, 'gt2048': gt2048, 'lab': lab, 'mask': mask, 
                'fg_mask': fg_mask, 'bg_mask': bg_mask, 'bg_mask_ori': bg_mask_ori, 
                'car_colormap': car_colormap, 'person_colormap': person_colormap, 
                'car_g_mask': car_g_mask, 'person_g_mask': person_g_mask, 
                'car_gt_crop': car_gt_crop, 'car_gt_crop2048': car_gt_crop2048, 'car_masked_img_crop': car_masked_img_crop, 'car_lab_crop': car_lab_crop, 
                'car_ints_mask_crop': car_ints_mask_crop, 'car_mask_crop': car_mask_crop, 'car_inst_cors': car_inst_cors, 
                'person_gt_crop': person_gt_crop, 'person_gt_crop2048': person_gt_crop2048, 'person_masked_img_crop': person_masked_img_crop, 'person_lab_crop': person_lab_crop, 
                'person_ints_mask_crop': person_ints_mask_crop, 'person_mask_crop': person_mask_crop, 'person_inst_cors': person_inst_cors, 
                'inst_map': inst_map, 'bg_coo_map': bg_coo_map, 
                'name': name}
                
        
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.name_list)

    def load_center_mask(self, crop_size, split):
        # rect
        height, width = crop_size, crop_size
        mask = np.ones((height, width), np.float32)
        if split == 'test':
            mask[crop_size:192, crop_size:192] = 0.
            w1 = crop_size
            w2 = crop_size + 128
            h1 = crop_size
            h2 = crop_size + 128
        else:
            w1 = random.randint(32, 96)
            w2 = w1 + 128
            h1 = random.randint(32, 96)
            h2 = h1 + 128
            mask[h1:h2, w1:w2] = 0.  # masked region=1, otherwise=0
        mask = mask.reshape((1,) + mask.shape).astype(np.float32)

        return mask

    def load_inpainting_mask(self, crop_size, split):   
        # rect
        height, width = crop_size, crop_size
        mask = np.zeros((height, width), np.float32)
        if split == 'test':
            mask[crop_size:192, crop_size:192] = 1.
            w1 = crop_size
            w2 = crop_size + 128
            h1 = crop_size
            h2 = crop_size + 128
        else:
            w1 = random.randint(32, 96)
            w = random.randint(120, 150)
            w2 = w1 + w
            
            h1 = random.randint(32, 96)
            h = random.randint(120, 150)
            h2 = h1 + h
            mask[h1:h2, w1:w2] = 1.  # masked region=1, otherwise=0
        mask = mask.reshape((1,) + mask.shape).astype(np.float32)

        return mask

    def load_right_mask(self, img_shapes, mask_rate=0.5):
        height, width = img_shapes, img_shapes
        mask = np.zeros((height, width), np.float32)

        mask_length = int(width * mask_rate)  # masked length
        w1 = width - mask_length
        mask[:, w1:] = 1.  # masked region=1, otherwise=0
        mask = mask.reshape((1,) + mask.shape).astype(np.float32)

        return mask

    def load_inst_mask(self, img_shapes, lab, fg_classes_list, bg_classes_list, fg_inst_mask_list):
        height, width = img_shapes, img_shapes
        mask = np.zeros((height, width), np.float32)
        
        for inst in fg_inst_mask_list:
            if np.sum(inst) > 400:
                curr_mask = np.zeros((256, 256), np.float32)
                ys,xs = np.where(inst > 0.)
                ymin, ymax, xmin, xmax = ys.min(), ys.max(), xs.min(), xs.max()
                mm = 5
                ymin = max(0, ymin-mm)
                ymax = min(255, ymax+mm)
                xmin = max(0, xmin-mm)
                xmax = min(255, xmax+mm)
                curr_mask[ymin:ymax, xmin:xmax] = 1.
                mask = mask + curr_mask * (1. - mask)
        
        mask = mask.reshape((1,) + mask.shape).astype(np.float32)

        return mask

    def load_seam_mask(self, img_shapes, box):
        m = 16
        height, width = img_shapes, img_shapes
        mask1 = np.ones((height, width), np.float32)
        mask2 = np.zeros((height, width), np.float32)
        
        mask1[box[0]+m:box[1]-m, box[2]+m:box[3]-m] = 0.
        mask2[box[0]-m:box[1]+m, box[2]-m:box[3]+m] = 1.
        
        mask = mask1 * mask2
        mask = mask.reshape((1,) + mask.shape).astype(np.float32)

        return torch.from_numpy(mask)


    # crop all instances
    def inst_crop(self, gt, gt2048, masked_img, lab, mask, inst_mask_list, class_list, max_num, class_id, min_size, crop_size, sum_min_ratio, ratio_range):
        
        size= crop_size
        
        img_crop_list = []
        img_crop2048_list = []
        masked_img_crop_list = []
        lab_crop_list = []
        inst_crop_list = []
        mask_crop_list = []
        
        g_mask_inst, cors_list = self.select_inst_cor(inst_mask_list, class_list, mask[0], class_id, min_size, sum_min_ratio, ratio_range)
        
        assert len(cors_list) < max_num
        assert len(cors_list) == len(g_mask_inst)
        
        for idx in range(len(cors_list)):
            curr_cor = cors_list[idx]
            curr_cor2048 = [min(c*4, 2047) for c in curr_cor]
            
            gt_crop = gt[:, curr_cor[0]:curr_cor[1], curr_cor[2]:curr_cor[3]].unsqueeze(0)
            gt_crop = F.interpolate(gt_crop, size, mode='bilinear')  # nearest, bilinear
            img_crop_list.append(gt_crop)
            
            gt_crop = gt2048[:, curr_cor2048[0]:curr_cor2048[1], curr_cor2048[2]:curr_cor2048[3]].unsqueeze(0)
            gt_crop = F.interpolate(gt_crop, size, mode='bilinear')  # nearest, bilinear
            img_crop2048_list.append(gt_crop)
            
            masked_img_crop = masked_img[:, curr_cor[0]:curr_cor[1], curr_cor[2]:curr_cor[3]].unsqueeze(0)
            masked_img_crop = F.interpolate(masked_img_crop, size, mode='bilinear')
            masked_img_crop_list.append(masked_img_crop)

            lab_crop = lab[:, curr_cor[0]:curr_cor[1], curr_cor[2]:curr_cor[3]].unsqueeze(0)
            lab_crop = F.interpolate(lab_crop, size, mode='nearest')
            lab_crop_list.append(lab_crop)
            
            curr_ints = torch.from_numpy(g_mask_inst[idx]).unsqueeze(0)
            ints_mask_crop = curr_ints[:, curr_cor[0]:curr_cor[1], curr_cor[2]:curr_cor[3]].unsqueeze(0)
            ints_mask_crop = F.interpolate(ints_mask_crop, size, mode='bilinear')  # nearest
            ints_mask_crop = torch.gt(ints_mask_crop, 0.5)
            inst_crop_list.append(ints_mask_crop)

            mask_crop = mask[:, curr_cor[0]:curr_cor[1], curr_cor[2]:curr_cor[3]].unsqueeze(0)
            mask_crop = F.interpolate(mask_crop, size, mode='nearest')
            mask_crop_list.append(mask_crop)
        
        gt_crop = torch.zeros((max_num,3,crop_size[0],crop_size[1]))
        gt_crop2048 = torch.zeros((max_num,3,crop_size[0],crop_size[1]))
        masked_img_crop = torch.zeros((max_num,3,crop_size[0],crop_size[1]))
        lab_crop = torch.zeros((max_num,1,crop_size[0],crop_size[1]))
        ints_mask_crop = torch.zeros((max_num,1,crop_size[0],crop_size[1]))
        mask_crop = torch.zeros((max_num,1,crop_size[0],crop_size[1]))
        cors = np.array([[0,0,0,0]] * max_num)
        if len(cors_list) == 0:
            g_mask_inst = [np.zeros((256,256))]

        
        for n in range(len(cors_list)):
            gt_crop[n] = img_crop_list[n]
            gt_crop2048[n] = img_crop2048_list[n]
            masked_img_crop[n] = masked_img_crop_list[n]
            lab_crop[n] = lab_crop_list[n]
            ints_mask_crop[n] = inst_crop_list[n]
            mask_crop[n] = mask_crop_list[n]
            cors[n] = np.array(cors_list[n])
            
        return gt_crop, gt_crop2048, masked_img_crop, lab_crop, ints_mask_crop, mask_crop, cors, g_mask_inst
            

    # select the necessary instances
    # class_id: class id of the instance
    # min_size: minimum size
    # sum_min_ratio: minimum proportion     SUM(instance) / SUM(bbox)
    # ratio_range: Aspect ratio range of the cropped instance images
    def select_inst_cor(self, inst_mask_list, class_list, mask, class_id, min_size, sum_min_ratio, ratio_range):
        g_mask_inst = []
        cors_list = []
        
        mask = mask.numpy()
        
        for i, ints in enumerate(inst_mask_list):
            ints_masked = ints * mask
            if class_list[i] == class_id:
                cor = self.gen_inst_bbox(ints, min_size=min_size)
                if sum(cor) > 0:
                    x_len = cor[1]-cor[0]
                    y_len = cor[3]-cor[2]
                    sum_ratio = np.sum(ints) / (x_len*y_len)
                    r_curr = x_len / y_len  # h/w
                else:
                    sum_ratio = 0.
                if not ratio_range:
                    ratio_range = (0, 100)
                
                if sum(cor) > 0 and np.sum(ints_masked) / np.sum(ints) > 0.4 and sum_ratio > sum_min_ratio:
                    if r_curr > ratio_range[0] and r_curr < ratio_range[1]:
                        g_mask_inst.append(ints)
                        cors_list.append(cor)
            else:
                pass
        
        assert len(g_mask_inst) == len(cors_list)
        
        return g_mask_inst, cors_list

    
    # calculate the coordinates of the boundary box (bbox) enclosing the instance
    def gen_inst_bbox(self, inst_mask, min_size):
        assert np.sum(inst_mask) > 0
        short_min, long_min = min_size
        
        x, y = np.where(inst_mask > 0.)
        x_min, x_max, y_min, y_max = np.min(x), np.max(x), np.min(y), np.max(y)
        x_len = x_max - x_min
        y_len = y_max - y_min
        short = min(x_len, y_len)
        long = max(x_len, y_len)
        m = 2
        
        if short > short_min and long > long_min:
            x_min = max(0, x_min-m)
            x_max = min(x_max+m, 255)
            y_min = max(0, y_min-m)
            y_max = min(y_max+m, 255)
            return [x_min, x_max, y_min, y_max]
        else:
            return [0, 0, 0, 0]

    
    def mask_binary(self, mask):
        b_mask = np.array(np.greater(mask, 0.5).astype(np.float32))
        return b_mask
    
def get_transform(img, normalize=True):
    transform_list = []

    transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)(img)

def brush_stroke_mask(H=256, W=256):
    min_num_vertex = 4
    max_num_vertex = 8
    mean_angle = 2 * math.pi / 5
    angle_range = 2 * math.pi / 15
    min_width = 50
    max_width = 140

    average_radius = math.sqrt(H * H + W * W) / 8
    mask = Image.new('L', (W, H), 0)

    num_vertex = random.randint(min_num_vertex, max_num_vertex)
    angle_min = mean_angle - random.uniform(0, angle_range)
    angle_max = mean_angle + random.uniform(0, angle_range)
    angles = []
    vertex = []
    for i in range(num_vertex):
        if i % 2 == 0:
            angles.append(2 * math.pi - random.uniform(angle_min, angle_max))
        else:
            angles.append(random.uniform(angle_min, angle_max))

    h, w = mask.size
    vertex.append((int(random.randint(0, w)), int(random.randint(0, h))))
    for i in range(num_vertex):
        r = np.clip(
            np.random.normal(loc=average_radius, scale=average_radius // 2),
            0, 2 * average_radius)
        new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
        new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
        vertex.append((int(new_x), int(new_y)))

    draw = ImageDraw.Draw(mask)
    width = int(random.uniform(min_width, max_width))
    draw.line(vertex, fill=1, width=width)
    for v in vertex:
        draw.ellipse((v[0] - width // 2,
                      v[1] - width // 2,
                      v[0] + width // 2,
                      v[1] + width // 2),
                     fill=1)

    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)
    
    mask = np.asarray(mask, np.float32)

    return mask

def get_mask_edge(mask):
    edge = cv2.Canny(mask, 0, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(20, 20))
    edge_mask = cv2.dilate(edge,kernel)
    
    return edge_mask


def inst2color(inst_map, max_idx=100, size=(256, 256)):
    cmap = np.array([[120, 120, 120],[180, 120, 120],[6, 230, 230],[80, 50, 50],[4, 200, 3],[120, 120, 80],[140, 140, 140],
            [204, 5, 255],[230, 230, 230],[4, 250, 7],[224, 5, 255],[235, 255, 7],[150, 5, 61],[120, 120, 70],[8, 255, 51],
            [255, 6, 82],[143, 255, 140],[204, 255, 4],[255, 51, 7],[204, 70, 3],[0, 102, 200],[61, 230, 250],[255, 6, 51],
            [11, 102, 255],[255, 7, 71],[255, 9, 224],[9, 7, 230],[220, 220, 220],[255, 9, 92],[112, 9, 255],[8, 255, 214],
            [7, 255, 224],[255, 184, 6],[10, 255, 71],[107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0],
            [255, 41, 10],[7, 255, 255],[224, 255, 8],[102, 8, 255],[255, 61, 6],[255, 194, 7],[255, 122, 8],[0, 255, 20],[255, 8, 41],
            [255, 5, 153],[6, 51, 255],[235, 12, 255],[160, 150, 20],[0, 163, 255],[140, 140, 140],[250, 10, 15],[20, 255, 0],
            [31, 255, 0],[255, 31, 0],[255, 224, 0],[153, 255, 0],[0, 0, 255],[255, 71, 0],[0, 235, 255],[0, 173, 255],[31, 0, 255],
            [11, 200, 200],[255, 82, 0],[0, 255, 245],[0, 61, 255],[0, 255, 112],[0, 255, 133],[255, 0, 0],[255, 163, 0],
            [255, 102, 0],[194, 255, 0],[0, 143, 255],[51, 255, 0],[0, 82, 255],[0, 255, 41],[0, 255, 173],[10, 0, 255],
            [173, 255, 0],[0, 255, 153],[255, 92, 0],[255, 0, 255],[255, 0, 245],[255, 0, 102],[255, 173, 0],[255, 0, 20],
            [255, 184, 184],[0, 31, 255],[0, 255, 61],[0, 71, 255],[255, 0, 204],[0, 255, 194],[0, 255, 82],[0, 10, 255],
            [0, 112, 255],[51, 0, 255],[0, 194, 255],[0, 122, 255],[0, 255, 163],[255, 153, 0],[0, 255, 10],[214, 255, 0],
            [0, 204, 255],[20, 0, 255],[255, 255, 0],[0, 153, 255],[0, 41, 255],[0, 255, 204],[41, 0, 255],[41, 255, 0],[173, 0, 255],
            [0, 245, 255],[71, 0, 255],[122, 0, 255],[0, 255, 184],[0, 92, 255],[184, 255, 0],[0, 133, 255],[255, 214, 0],
            [25, 194, 194],[102, 255, 0],[92, 0, 255],],dtype=np.uint8)
    color = np.zeros((3, size[0], size[1]))
    for i in range(1, max_idx):
        mask = (inst_map.astype(np.int64) == i)
        color[0][mask] = cmap[i][0]
        color[1][mask] = cmap[i][1]
        color[2][mask] = cmap[i][2]
    color = np.transpose(color, (1, 2, 0))
    return color