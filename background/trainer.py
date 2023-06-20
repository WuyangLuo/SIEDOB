from networks.bg_G import bg_G
from networks.SNPatchDiscriminator import SNPatchDiscriminator
from utils import get_scheduler, weights_init, save_network, save_latest_network, get_model_list
import torch
import torch.nn as nn
import torch.nn.functional as F
import loss
import os
import numpy as np
import cv2
import random

class Trainer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # setting basic params
        self.cfg = cfg
        if self.cfg['is_train']:
            self.model_names = ['netG_BG', 'netD_BG', 'netD_patch']
        else:  # during test time, only load G
            self.model_names = ['netG']

        # Initiate the submodules and initialization params
        self.netG_BG = bg_G(self.cfg)
        self.netG_BG.apply(weights_init('gaussian'))
        
        self.netD_BG = SNPatchDiscriminator(self.cfg)
        self.netD_BG.apply(weights_init('gaussian'))
        
        self.netD_patch = SNPatchDiscriminator(self.cfg, num_layer=6)
        self.netD_patch.apply(weights_init('gaussian'))

        self.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if torch.cuda.is_available() \
            else torch.ByteTensor

        # Setup the optimizers and schedulers
        if cfg['is_train']:
            lr = self.cfg['lr']
            beta1 = self.cfg['beta1']
            beta2 = self.cfg['beta2']
            
            # set optimizers
            self.optimizers = []
            G_BG_params = list(self.netG_BG.parameters())
            self.optimizer_G_BG = torch.optim.Adam(G_BG_params, lr=lr, betas=(beta1, beta2))
            self.optimizers.append(self.optimizer_G_BG)
            
            D_bg_params = list(self.netD_BG.parameters())
            self.optimizer_D_bg = torch.optim.Adam(D_bg_params, lr=lr, betas=(beta1, beta2))
            self.optimizers.append(self.optimizer_D_bg)

            D_patch_params = list(self.netD_patch.parameters())
            self.optimizer_D_patch = torch.optim.Adam(D_patch_params, lr=lr, betas=(beta1, beta2))
            self.optimizers.append(self.optimizer_D_patch)
            
            self.opt_names = ['optimizer_G_BG', 'optimizer_D_bg', 'optimizer_D_patch']

            # set schedulers
            self.schedulers = [get_scheduler(optimizer, self.cfg) for optimizer in self.optimizers]
            # set criterion
            self.criterionGAN_global = loss.GANLoss(cfg['gan_mode']).cuda()
            self.criterionGAN_patch = loss.GANLoss_MultiD(cfg['gan_mode'], tensor=self.FloatTensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionVGG = loss.VGGLoss()
            self.criterionFeat = torch.nn.L1Loss()
        
        # Losses
        self.G_BG_losses = {}
        self.D_BG_losses = {}
        self.D_patch_losses = {}

        self.Losses_name = ['G_BG_losses', 'D_BG_losses', 'D_patch_losses']

######################################################################################
    def set_input(self, input):
        self.masked_img = input['masked_img'].cuda()     # mask image
        self.gt = input['img'].cuda()        # real image
        self.gt2048 = input['gt2048'].cuda()
        self.mask = input['mask'].cuda()    # mask image
        self.lab = input['lab'].long().cuda()  # label image scatter_ require .long() type
        self.segmap = self.scatter_lab(self.lab, self.cfg['lab_dim'])

        self.fg_mask = input['fg_mask'].cuda()
        self.bg_mask = input['bg_mask'].cuda()
        self.bg_mask_ori = input['bg_mask_ori'].cuda()
        
        self.car_colormap = input['car_colormap'].cuda()
        self.person_colormap = input['person_colormap'].cuda()
        
        self.car_g_mask = input['car_g_mask'].cuda()
        self.person_g_mask = input['person_g_mask'].cuda()
        
        self.car_gt_crop = self.mask_crop_batch(input['car_gt_crop'].cuda())
        self.car_gt_crop2048 = self.mask_crop_batch(input['car_gt_crop2048'].cuda())
        self.car_masked_img_crop = self.mask_crop_batch(input['car_masked_img_crop'].cuda())
        self.car_ints_mask_crop = self.mask_crop_batch(input['car_ints_mask_crop'].cuda())
        self.car_mask_crop = self.mask_crop_batch(input['car_mask_crop'].cuda())
        self.car_inst_cors = self.mask_crop_batch(input['car_inst_cors'])

        self.person_gt_crop = self.mask_crop_batch(input['person_gt_crop'].cuda())
        self.person_gt_crop2048 = self.mask_crop_batch(input['person_gt_crop2048'].cuda())
        self.person_masked_img_crop = self.mask_crop_batch(input['person_masked_img_crop'].cuda())
        self.person_ints_mask_crop = self.mask_crop_batch(input['person_ints_mask_crop'].cuda())
        self.person_mask_crop = self.mask_crop_batch(input['person_mask_crop'].cuda())
        self.person_inst_cors = self.mask_crop_batch(input['person_inst_cors'])
        
        self.inst_map = input['inst_map'].cuda()
        self.edge_map = self.get_edges(self.inst_map)
        self.edge_map = self.edge_map * self.mask
        
        self.segmap_edge = torch.cat((self.segmap, self.edge_map), dim=1)

        self.name = input['name']
        
        self.bg_segmap = self.segmap_edge * self.bg_mask
        self.gt_bg_mask = self.gt * self.bg_mask
        self.masked_img = self.masked_img * self.bg_mask
        self.mask = self.mask * self.bg_mask
        
        self.bg_coo_map = input['bg_coo_map']
        # print(self.bg_coo_map.size())  # torch.Size([5, 35, 256, 256])

    # create one-hot label map
    def scatter_lab(self, lab, c_num):
        lab_map = lab
        bs, _, h, w = lab_map.size()
        input_label = self.FloatTensor(bs, c_num, h, w).zero_()
        segmap = input_label.scatter_(1, lab_map, 1.0)
        return segmap
    
    def mask_crop_batch(self, crop):
        list = []
        for n in range(crop.size()[0]):
            list.append(crop[n])
        batch = torch.cat(list, dim=0)
        return batch

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        
        return edge.float()

    def forward(self, train=True):
        self.bg_output = self.netG_BG(torch.cat((self.masked_img, self.bg_segmap, self.mask), dim=1), self.segmap, self.mask, self.bg_mask)
        self.masked_fake = self.bg_output * (self.bg_mask * self.mask) + self.gt * (1. - self.mask * self.bg_mask)
        
        if train:
            self.real_patch, self.fake_patch = self.crop_patch(self.gt, self.masked_fake, self.mask, self.segmap)

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        
        # Fake
        fake = torch.cat([self.segmap, self.masked_fake.detach()], dim=1)
        pred_fake = self.netD_BG(fake)
        self.D_BG_losses['loss_D_fake_BG'] = self.criterionGAN_global(pred_fake, False, for_discriminator=True) * self.cfg['lambda_gan']
        # Real
        real = torch.cat([self.segmap, self.gt], dim=1)
        pred_real = self.netD_BG(real)
        self.D_BG_losses['loss_D_real_BG'] = self.criterionGAN_global(pred_real, True, for_discriminator=True) * self.cfg['lambda_gan']

    def compute_Dpatch_loss(self):
        # Fake
        fake = self.fake_patch.detach()
        pred_fake = self.netD_patch(fake)
        self.D_patch_losses['loss_Dpatch_fake'] = self.criterionGAN_global(pred_fake, False, for_discriminator=True) * self.cfg['lambda_patch']
        # Real
        real = self.real_patch
        pred_real = self.netD_patch(real)
        self.D_patch_losses['loss_Dpatch_real'] = self.criterionGAN_global(pred_real, True, for_discriminator=True) * self.cfg['lambda_patch']

    def compute_G_loss(self):
        """Calculate losses for the generator"""
        self.G_BG_losses['L1_bg'] = torch.mean(torch.abs(self.bg_output - self.gt_bg_mask)) * self.cfg['lambda_L1']
        
        # GAN loss
        # bg
        fake_global = torch.cat([self.segmap, self.masked_fake], dim=1)
        pred_fake_global = self.netD_BG(fake_global)
        self.G_BG_losses['G_GAN_bg'] = self.criterionGAN_global(pred_fake_global, True, for_discriminator=False) * self.cfg['lambda_gan']

        fake_global = self.fake_patch
        pred_fake_global = self.netD_patch(fake_global)
        self.G_BG_losses['G_GAN_Dpatch'] = self.criterionGAN_global(pred_fake_global, True, for_discriminator=False) * self.cfg['lambda_patch']

        # VGG loss
        if not self.cfg['no_vgg_loss']:
            self.G_BG_losses['VGG_bg'] = self.criterionVGG(self.bg_output, self.gt_bg_mask) * self.cfg['lambda_vgg']


    def optimize_parameters(self):
        self.forward()

        # update global D
        self.set_requires_grad(self.netD_BG, True)
        self.optimizer_D_bg.zero_grad()
        self.compute_D_loss()
        d_bg_loss = sum(self.D_BG_losses.values()).mean()
        d_bg_loss.backward()
        self.optimizer_D_bg.step()

        # update patch D
        self.set_requires_grad(self.netD_patch, True)
        self.optimizer_D_patch.zero_grad()
        self.compute_Dpatch_loss()
        d_patch_loss = sum(self.D_patch_losses.values()).mean()
        d_patch_loss.backward()
        self.optimizer_D_patch.step()
        
        # update G
        self.set_requires_grad(self.netD_BG, False)
        self.set_requires_grad(self.netD_patch, False)
        self.optimizer_G_BG.zero_grad()        # set G's gradients to zero
        self.compute_G_loss()
        g_loss = sum(self.G_BG_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G_BG.step()

    #########################################################################################################
    ########## util func #############
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def visual_results(self):
        return {'masked_img': self.masked_img, 'gt': self.gt, 'lab': self.lab, 'mask': self.mask, 'gt2048': self.gt2048, 
                'fg_mask': self.fg_mask, 'bg_mask': self.bg_mask, 'bg_mask_ori': self.bg_mask_ori, 'edge_map': self.edge_map, 
                'gt_bg_mask': self.gt_bg_mask, 
                'bg_output': self.bg_output, 'masked_fake': self.masked_fake, 
                'real_patch': self.real_patch[:,:3], 'fake_patch': self.fake_patch[:,:3],
                }, self.name

    def print_losses(self):
        for name in self.Losses_name:
            loss = getattr(self, name)
            print('===== ', name, ' =====')
            for v,k in loss.items():
                print(v, ': ', k)


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def save_nets(self, epoch, cfg, suffix=''):
        save_file = {}
        save_file['epoch'] = epoch
        for name in self.model_names:
            net = getattr(self, name)
            save_file[name] = net.cpu().state_dict()
            if torch.cuda.is_available():
                net.cuda()
        for name in self.opt_names:
            opt = getattr(self, name)
            save_file[name] = opt.state_dict()
        save_filename = '%03d_ckpt_%s.pth' % (epoch, suffix)
        save_path = os.path.join(cfg['checkpoints_dir'], save_filename)
        torch.save(save_file, save_path)

    def save_latest_nets(self, epoch, cfg):
        save_file = {}
        save_file['epoch'] = epoch
        for name in self.model_names:
            net = getattr(self, name)
            save_file[name] = net.cpu().state_dict()
            if torch.cuda.is_available():
                net.cuda()
        for name in self.opt_names:
            opt = getattr(self, name)
            save_file[name] = opt.state_dict()
        save_filename = 'latest_ckpt.pth'
        save_path = os.path.join(cfg['checkpoints_dir'], save_filename)
        torch.save(save_file, save_path)

    def print_networks(self):
        """Print the total number of parameters in the network and network architecture"""
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                # print network architecture
                print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def inst_crop(self, image, cors, size, inst_masks):
        crops = []
        crop_num = 0
        for n in range(image.size()[0]):  # batch size
            cor_list = cors[n]
            for inst_i in range(cor_list.size()[0]):
                if torch.sum(cor_list[inst_i]) > 0:
                    # print(image.size())
                    # print(inst_masks.size())
                    curr_image = image[n:n+1, :, :, :] * inst_masks[n:n+1, inst_i:inst_i+1, :, :]
                    curr_crop = curr_image[:, :, cor_list[inst_i][0]:cor_list[inst_i][1], cor_list[inst_i][2]:cor_list[inst_i][3]]
                    curr_crop = F.interpolate(curr_crop, size, mode='bilinear')
                    crops.append(curr_crop)
                    crop_num += 1
        if crop_num >= 1:
            crops = torch.cat(crops, dim=0)
        elif crop_num == 0:
            _, c, _, _ = image.size()
            crops = torch.zeros((1, c, size[0], size[1]))
        
        return crops, crop_num

    def resume(self, checkpoint_dir, ckpt_filename=None):
        if not ckpt_filename:
            ckpt_filename = 'latest_ckpt.pth'
        ckpt = torch.load(os.path.join(checkpoint_dir, ckpt_filename))
        cur_epoch = ckpt['epoch']
        for name in self.model_names:
            net = getattr(self, name)
            net.load_state_dict(ckpt[name])
            print('load model %s of epoch %d' % (name, cur_epoch))
        for name in self.opt_names:
            opt = getattr(self, name)
            opt.load_state_dict(ckpt[name])
            print('load opt %s of epoch %d' % (name, cur_epoch))
        return cur_epoch



    ### Dpatch ###
    def edge_mask(self, mask):
        mask = mask.cpu().float().numpy().astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        erode = cv2.erode(mask, kernel)
        
        edge = mask * (1. - erode)

        return edge

    def patchify_image(self, img, xc, yc, size=32):
        size = int(size/2)
        if xc < size:
            x_l = 0
            x_r = size*2
        elif xc + size >= 255 :
            x_l = 255 - size*2
            x_r = 255
        else:
            x_l = xc - size
            x_r = xc + size

        if yc < size:
            y_l = 0
            y_r = size*2
        elif yc + size >= 255 :
            y_l = 255 - size*2
            y_r = 255
        else:
            y_l = yc - size
            y_r = yc + size
        
        return F.interpolate(img[:, :, x_l:x_r, y_l:y_r], (160,160), mode='bilinear')
        
    def crop_patch(self, real, fake, mask, segmap, patch_num=4):
        B, _, _, _= real.size()
        real_patch_list = []
        fake_patch_list = []
        
        for b in range(B):
            curr_mask = mask[b,0,:,:]
            edge = self.edge_mask(curr_mask)
            for n in range(patch_num):
                if np.sum(edge) > 0:
                    xl, yl = np.where(edge == 1)
                    c = random.randrange(0, xl.shape[0])
                    xc, yc = xl[c], yl[c]
                    curr_size = random.randint(96,160)
                    real_p = self.patchify_image(real[b:b+1], xc, yc, curr_size)
                    fake_p = self.patchify_image(fake[b:b+1], xc, yc, curr_size)
                    seg_p = self.patchify_image(segmap[b:b+1], xc, yc, curr_size)
                    real_patch_list.append(torch.cat((real_p, seg_p), dim=1))
                    fake_patch_list.append(torch.cat((fake_p, seg_p), dim=1))
        
        real_patch = torch.cat((real_patch_list), dim=0)
        fake_patch = torch.cat((fake_patch_list), dim=0)
        
        return real_patch, fake_patch
