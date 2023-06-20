from utils import *
import argparse
import numpy as np
import datetime
from trainer import Trainer
from dataset import SIE_Dataset
import torch
from torch.utils.data import DataLoader
import os
import shutil
import cv2
from metrics.fid_score import calculate_fid_given_paths


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--dataset_name', type=str, default='cityscapes512x256', help="dataset name")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--resume_dir', type=str, default='', help="outputs path")
opts = parser.parse_args()

print_options(opts)


# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Load experiment setting
cfg = get_config(opts.config)
# datasets setting
if opts.dataset_name == 'cityscapes512x256':
    cfg['lab_dim'] = 34
    cfg['max_epoch'] = 500
    cfg['epoch_init_lr'] = 400
    cfg['niter_decay'] = 100
    cfg['test_freq'] = 20


trainer = Trainer(cfg)
trainer.cuda()

# print model information
trainer.print_networks()

ckt_path = 'submodels/Fnet.pth'
state_dict = torch.load(ckt_path)['netG_F']
trainer.netG_F.load_state_dict(state_dict)



# Setup dataset
dataset_root = 'example/'
test_dataset = SIE_Dataset(cfg, dataset_root, split='test', dataset_name=opts.dataset_name, mask_type='out_dis')
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)
print('test dataset containing ', len(test_loader), 'images')

# Start training
for i, data in enumerate(test_loader):  # inner loop within one epoch
    trainer.eval()
    trainer.set_input(data)  # unpack data from dataset and apply preprocessing
    trainer.forward()
    
    ##############################################################################################################
    results, img_name = trainer.visual_results()

    save_dir = os.path.join('results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    cv2.imwrite(os.path.join(save_dir, img_name[0]+'.png'), tensor2im(results['masked_F']))

