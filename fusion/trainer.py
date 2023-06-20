from networks.G_stage import G_stage
from networks.F_stage import F_stage
from networks.SNPatchDiscriminator import SNPatchDiscriminator
from utils import get_scheduler, weights_init, save_network, save_latest_network, get_model_list
import torch
import torch.nn as nn
import torch.nn.functional as F
import loss
import os

class Trainer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # setting basic params
        self.cfg = cfg
        self.mask_rate = self.cfg['mask_rate']
        if self.cfg['is_train']:
            self.model_names = ['netG_F', 'netD_F']
        else:  # during test time, only load G
            self.model_names = ['netG']
        
        self.G_stage = G_stage(self.cfg)
        self.set_requires_grad(self.G_stage, False)

        # Initiate the submodules and initialization params
        self.netG_F = F_stage(self.cfg)
        self.netD_F = SNPatchDiscriminator(self.cfg, in_dim=3)

        self.netG_F.apply(weights_init('gaussian'))
        self.netD_F.apply(weights_init('gaussian'))

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
            G_F_params = list(self.netG_F.parameters())
            self.optimizer_G_F = torch.optim.Adam(G_F_params, lr=lr, betas=(beta1, beta2))
            self.optimizers.append(self.optimizer_G_F)
            
            D_F_params = list(self.netD_F.parameters())
            self.optimizer_D_F = torch.optim.Adam(D_F_params, lr=lr, betas=(beta1, beta2))
            self.optimizers.append(self.optimizer_D_F)
            
            self.opt_names = ['optimizer_G_F', 'optimizer_D_F']

            # set schedulers
            self.schedulers = [get_scheduler(optimizer, self.cfg) for optimizer in self.optimizers]
            # set criterion
            self.criterionGAN_global = loss.GANLoss(cfg['gan_mode']).cuda()
            self.criterionGAN_patch = loss.GANLoss_MultiD(cfg['gan_mode'], tensor=self.FloatTensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionVGG = loss.VGGLoss()
            self.criterionFeat = torch.nn.L1Loss()
        
        # Losses
        self.G_F_losses = {}
        self.D_F_losses = {}

        self.Losses_name = ['G_F_losses', 'D_F_losses']

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

        self.car_sty_images = self.mask_crop_batch(input['car_sty_images'].cuda())
        self.car_sty_inst_images = self.mask_crop_batch(input['car_sty_inst_images'].cuda())
        self.person_sty_images = self.mask_crop_batch(input['person_sty_images'].cuda())
        self.person_sty_inst_images = self.mask_crop_batch(input['person_sty_inst_images'].cuda())

        self.name = input['name']

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

    def forward(self):
        
        with torch.no_grad():
            inst_output_list, self.fg_output, self.bg_output \
                = self.G_stage(self.gt, self.masked_img, self.mask, self.fg_mask,
                            self.car_gt_crop, self.car_gt_crop2048, self.car_masked_img_crop, self.car_ints_mask_crop, self.car_mask_crop, self.car_inst_cors, 
                            self.person_gt_crop, self.person_gt_crop2048, self.person_masked_img_crop, self.person_ints_mask_crop, self.person_mask_crop, self.person_inst_cors, 
                            self.segmap_edge, self.segmap, self.bg_mask,
                            self.car_sty_images, self.car_sty_inst_images, self.person_sty_images, self.person_sty_inst_images)

            self.car_input_filtering, self.car_output, self.car_inst_mask_filtering, self.car_mask_filtering, self.car_num, \
            self.person_input_filtering, self.person_output, self.person_inst_mask_filtering, self.person_mask_filtering, self.person_num, \
            self.embed_car_output, self.embed_person_output, self.embed_output, self.gt_fg_gen = inst_output_list
            
            self.masked_G = self.fg_output * (self.fg_mask * self.mask) + self.bg_output * (self.bg_mask * self.mask) + self.gt * (1. - self.mask)
        
        self.fake_F = self.netG_F(self.masked_G, self.segmap, self.mask, self.edge_map)
        self.masked_F = self.fake_F * self.mask + self.gt * (1. - self.mask)

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        
        # Fake
        fake = self.masked_F.detach()
        pred_fake = self.netD_F(fake)
        self.D_F_losses['loss_D_fake_F'] = self.criterionGAN_global(pred_fake, False, for_discriminator=True) * self.cfg['lambda_gan']
        # Real
        real = self.gt
        pred_real = self.netD_F(real)
        self.D_F_losses['loss_D_real_F'] = self.criterionGAN_global(pred_real, True, for_discriminator=True) * self.cfg['lambda_gan']

    def compute_G_loss(self):
        """Calculate losses for the generator"""
        # self.G_F_losses['L1_bg'] = torch.mean(torch.abs(self.bg_output - self.gt)) * self.cfg['lambda_L1']
        
        # GAN loss
        # bg
        fake_global = self.masked_F
        pred_fake_global = self.netD_F(fake_global)
        self.G_F_losses['G_GAN_F'] = self.criterionGAN_global(pred_fake_global, True, for_discriminator=False) * self.cfg['lambda_gan']

        # VGG loss
        if not self.cfg['no_vgg_loss']:
            self.G_F_losses['VGG_bg'] = self.criterionVGG(self.fake_F, self.gt) * self.cfg['lambda_vgg']


    def optimize_parameters(self):
        self.forward()
        self.G_stage.eval()

        # update global D
        self.set_requires_grad(self.netD_F, True)
        self.optimizer_D_F.zero_grad()
        self.compute_D_loss()
        d_bg_loss = sum(self.D_F_losses.values()).mean()
        d_bg_loss.backward()
        self.optimizer_D_F.step()
        
        # update G
        # self.set_requires_grad(self.netD_patch, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD_F, False)
        self.optimizer_G_F.zero_grad()        # set G's gradients to zero
        self.compute_G_loss()
        g_bg_loss = sum(self.G_F_losses.values()).mean()
        g_bg_loss.backward()
        self.optimizer_G_F.step()

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
                'car_g_mask': self.car_g_mask, 'person_g_mask': self.person_g_mask,
                'fg_output': self.fg_output, 'bg_output': self.bg_output, 
                'car_input_filtering': self.car_input_filtering, 'car_output': self.car_output, 'car_mask_filtering': self.car_mask_filtering, 
                'car_inst_mask_filtering': self.car_inst_mask_filtering, 
                'person_input_filtering': self.person_input_filtering, 'person_output': self.person_output, 'person_mask_filtering': self.person_mask_filtering, 
                'person_inst_mask_filtering': self.person_inst_mask_filtering,
                'embed_car_output': self.embed_car_output, 'embed_person_output': self.embed_person_output, 'embed_output': self.embed_output,
                'gt_fg_gen': self.gt_fg_gen, 
                'car_gt_crop': self.car_gt_crop, 'car_gt_crop2048': self.car_gt_crop2048, 
                'person_gt_crop': self.person_gt_crop, 'person_gt_crop2048': self.person_gt_crop2048, 
                'masked_G': self.masked_G, 'fake_F': self.fake_F, 'masked_F': self.masked_F, 
                'car_sty_images': self.car_sty_images, 'car_sty_inst_images': self.car_sty_inst_images, 'person_sty_images': self.person_sty_images, 'person_sty_inst_images': self.person_sty_inst_images,
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