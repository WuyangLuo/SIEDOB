import os
import cv2
import math
import numpy as np
from torch.utils.data import Dataset
import os.path
import random
import torchvision.transforms as transforms
import torch
from utils import make_dataset
from PIL import Image, ImageDraw


class SIE_Dataset(Dataset):
    def __init__(self, cfg, dataset_root, split='train', mask_type='ff_mask', dataset_name=''):
        self.split = split
        self.cfg = cfg
        self.mask_type = mask_type
        self.sh, self.sw = 64, 64

        self.data = []

        if dataset_name == 'cityscapes512x256':
            class_set = [('car', torch.tensor([1,0,0])), ('person', torch.tensor([0,1,0]))]

        if self.split == 'train':
            for s in class_set:
                curr_d = []
                self.img = os.path.join(dataset_root, s[0], self.split, 'images')
                self.ins = os.path.join(dataset_root, s[0], self.split, 'inst_mask')
                name_list = os.listdir(self.img)
                for n in name_list :
                    if n.endswith('jpg'):
                        curr_d.append((os.path.join(self.img,n[:-4]+'.jpg'), os.path.join(self.ins,n[:-4]+'.png'), s[1], n[:-4]))
                
                curr_d = curr_d * 10
                self.data.extend(curr_d[:10000])
        
        elif self.split == 'test':
            for s in class_set:
                self.img = os.path.join(dataset_root, s[0], self.split, 'images')
                self.ins = os.path.join(dataset_root, s[0], self.split, 'inst_mask')
                name_list = os.listdir(self.img)
                for n in name_list :
                    if n.endswith('jpg'):
                        self.data.append((os.path.join(self.img,n[:-4]+'.jpg'), os.path.join(self.ins,n[:-4]+'.png'), s[1], n[:-4]))

            self.data.sort()
            self.mask_path = ''  # path to object mask
            mask_list = os.listdir(self.mask_path)
            mask_list.sort()
            self.mask_list = mask_list[:len(self.data)]

    def __getitem__(self, index):
        d = self.data[index]
        # input data
        name = d[3]
        img = cv2.imread(d[0])                     # (64, 32, 3)
        inst_mask = cv2.imread(d[1],0) / 255

        img = cv2.resize(img, (self.sh, self.sw))
        inst_mask = cv2.resize(inst_mask, (self.sh, self.sw), interpolation=cv2.INTER_NEAREST)
        
        if self.split == 'train':
            # resize
            if random.random() > 0.5:
                new_h, new_w = self.sh+8, self.sw+8
                img = cv2.resize(img, (new_h, new_w))
                inst_mask = cv2.resize(inst_mask, (new_h, new_w), interpolation=cv2.INTER_NEAREST)
                w, h, _ = img.shape  # 64, 32
                w_l = 0
                h_l = 0
                if w > self.sw:
                    w_l = random.randint(0, w - self.sw)
                if h > self.sh:
                    h_l = random.randint(0, h - self.sh)
                
                img = img[w_l:w_l+self.sw, h_l:h_l+self.sh]
                inst_mask = inst_mask[w_l:w_l+self.sw, h_l:h_l+self.sh]
        
            # flip
            if random.random() > 0.5:
                # print('filp')
                img = np.flip(img,axis=1).copy()
                inst_mask = np.flip(inst_mask,axis=1).copy()

        
        img = get_transform(img)
        inst_mask = get_transform(inst_mask, normalize=False)
        
        if self.split == 'train':
            mask_type = index % 3
            # 
            if mask_type < 2:
                mask = brush_stroke_mask(H=self.sh, W=self.sw)
                mask = mask.reshape((1,) + mask.shape).astype(np.float32)
            elif mask_type == 2:
                mask = self.load_right_mask(H=self.sh, W=self.sw)
            
            if mask_type <= 2:
                mask = torch.from_numpy(mask)
        
        else:
            mask = cv2.imread(os.path.join(self.mask_path, self.mask_list[index]), 0) / 255
            mask = cv2.resize(mask, (self.sh, self.sw), interpolation=cv2.INTER_NEAREST)
            mask = mask.reshape((1,) + mask.shape).astype(np.float32)
            mask = torch.from_numpy(mask)


        inst_mask = inst_mask.float()
        mask = mask.float()
        
        mask = mask * inst_mask
        masked_img = img * (1. - mask) * inst_mask
        gt = img
        img = img * inst_mask
        class_map = inst_mask * d[2].unsqueeze(-1).unsqueeze(-1)

        
        return {'img': img, 'class_map': class_map, 'gt': gt, 'masked_img': masked_img, 'inst_mask': inst_mask, 'mask': mask, 'name': name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.data)


    def load_right_mask(self, W=64, H=64):
        mask_rate = random.random()/2.0 + 0.15
        height, width = H, W
        mask = np.zeros((height, width), np.float32)

        w1 = width - int(width * mask_rate)
        h1 = height - int(height * mask_rate)
        if random.random() > 0.5:
            if random.random() > 0.5:
                mask[:, w1:] = 1.  # masked region=1，otherwise=0
            else:
                mask[:, :width-w1] = 1.
        else:
            if random.random() > 0.5:
                mask[h1:, :] = 1.
            else:
                mask[:height-h1, :] = 1.
        mask = mask.reshape((1,) + mask.shape).astype(np.float32)

        return mask


def get_transform(img, normalize=True):
    transform_list = []

    transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)(img)

def brush_stroke_mask(H=64, W=64):
    min_num_vertex = 3
    max_num_vertex = 6
    mean_angle = 2 * math.pi / 5
    angle_range = 2 * math.pi / 15
    min_width = 30
    max_width = 60

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