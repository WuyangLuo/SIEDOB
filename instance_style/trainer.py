from networks.generator import generator
from networks.SNPatchDiscriminator import SNPatchDiscriminator
from utils import get_scheduler, weights_init, save_network, save_latest_network, get_model_list
import torch
import torch.nn as nn
from torch.nn import functional as F
import loss
import os
from torch import autograd

class Trainer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # setting basic params
        self.cfg = cfg
        if self.cfg['is_train']:
            self.model_names = ['netG', 'netD_style']
        else:  # during test time, only load G
            self.model_names = ['G']

        # Initiate the submodules and initialization params
        self.netG = generator(self.cfg)
        self.netD_style = SNPatchDiscriminator(self.cfg, num_layer=6)

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
            
            G_params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(G_params, lr=lr, betas=(beta1, beta2))
            self.optimizers.append(self.optimizer_G)
            
            D_style_params = list(self.netD_style.parameters())
            self.optimizer_D_style = torch.optim.Adam(D_style_params, lr=lr, betas=(beta1, beta2))
            self.optimizers.append(self.optimizer_D_style)

            
            self.opt_names = ['optimizer_G', 'optimizer_D_style']
            # set schedulers
            self.schedulers = [get_scheduler(optimizer, self.cfg) for optimizer in self.optimizers]
            # set criterion
            self.criterionGAN_global = loss.GANLoss(cfg['gan_mode']).cuda()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionVGG = loss.VGGLoss()
            # self.criterionDiv = loss.DivLoss()
            self.criterionInner = loss.Inner_Loss()

        self.G_losses = {}
        self.D_losses = {}

######################################################################################
    def set_input(self, input):
        self.gt = input['gt'].cuda()
        self.class_map = input['class_map'].cuda()
        self.inst_mask = input['inst_mask'].cuda()

        self.sty_gt = input['sty_gt'].cuda()
        self.sty_class_map = input['sty_class_map'].cuda()
        self.sty_inst_mask = input['sty_inst_mask'].cuda()

        self.name = input['name']


    def forward(self):
        self.rec_fake, self.rec_sc, self.rec_cycle_sc_fake = self.netG(self.class_map, self.gt, self.class_map, self.inst_mask)
        self.masked_rec_fake = self.rec_fake * self.inst_mask

        self.sty_fake, self.sty_sc, self.sty_cycle_sc_fake = self.netG(self.class_map, self.sty_gt, self.sty_class_map, self.inst_mask)
        self.masked_sty_fake = self.sty_fake * self.inst_mask

        self.all_fake = torch.cat((self.rec_fake, self.sty_fake), dim=0)
        self.all_masked_fake = torch.cat((self.masked_rec_fake, self.masked_sty_fake), dim=0)
        self.all_cls_map = torch.cat((self.class_map, self.class_map), dim=0)
        self.all_sc = torch.cat((self.rec_sc, self.sty_sc), dim=0)
        self.all_cycle_sc = torch.cat((self.rec_cycle_sc_fake, self.sty_cycle_sc_fake), dim=0)
        self.all_gt = torch.cat((self.gt, self.gt), dim=0)



    def compute_D_style_loss(self):
        # Fake
        fake = torch.cat([self.all_masked_fake.detach(), self.all_cls_map], dim=1)
        pred_fake = self.netD_style(fake)
        self.D_losses['loss_D_fake'] = self.criterionGAN_global(pred_fake, False, for_discriminator=True) * self.cfg['lambda_gan']
        # Real
        real = torch.cat([self.all_gt, self.all_cls_map], dim=1)
        pred_real = self.netD_style(real)
        self.D_losses['loss_D_real'] = self.criterionGAN_global(pred_real, True, for_discriminator=True) * self.cfg['lambda_gan']
        
        return self.D_losses


    def compute_G_loss(self):
        """Calculate losses for the generator"""
        # # L1 loss
        self.G_losses['L1'] = torch.mean(torch.abs(self.rec_fake - self.gt)) * self.cfg['lambda_L1']

        fake = torch.cat([self.all_masked_fake, self.all_cls_map], dim=1)
        pred_fake_global = self.netD_style(fake)
        self.G_losses['G_GAN'] = self.criterionGAN_global(pred_fake_global, True, for_discriminator=False) * self.cfg['lambda_gan']

        # VGG loss
        if not self.cfg['no_vgg_loss']:
            self.G_losses['VGG'] = self.criterionVGG(self.all_fake, self.all_gt) * self.cfg['lambda_vgg']

        # style code cycle loss
        # self.G_losses['style_cycle'] = torch.mean(torch.abs(self.all_sc - self.all_cycle_sc)) * self.cfg['lambda_style']
        self.G_losses['style_cycle'] = self.criterionInner(self.all_sc, self.all_cycle_sc)

        return self.G_losses


    def optimize_parameters(self):
        self.forward()
        # update patch D
        self.set_requires_grad(self.netD_style, True)  # enable backprop for D
        self.optimizer_D_style.zero_grad()  # set D's gradients to zero
        d_losses = self.compute_D_style_loss()
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D_style.step()

        # update G
        self.set_requires_grad(self.netD_style, False)  # D requires no gradients when optimizing G
        # self.set_requires_grad(self.netD_mm, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.g_losses = self.compute_G_loss()
        g_loss = sum(self.g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()             # udpate G's weights

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
        return {'gt': self.gt, 'inst_mask': self.inst_mask, 'sty_gt': self.sty_gt, 'sty_inst_mask': self.sty_inst_mask,
                'masked_rec_fake': self.masked_rec_fake, 'masked_sty_fake': self.masked_sty_fake,
                'rec_fake': self.rec_fake, 'sty_fake': self.sty_fake}, self.name

    def print_losses(self):
        print('G Losses')
        for v,k in self.G_losses.items():
            print(v, ': ', k)
            
        print('D Losses')
        for v,k in self.D_losses.items():
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

    def d_r1_loss(self, real_pred, real_img):
        # with conv2d_gradfix.no_weight_gradients():
        
        grad_real, = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

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