import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torchvision

# ==================================== SPADE =========================================
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        ks = 3

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1. + gamma) + beta

        return out

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, lab_nc, dilation=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        padding = dilation
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=padding, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=padding, dilation=dilation)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # # apply spectral norm if specified
        # if 'spectral' in norm_G:
        #     self.conv_0 = spectral_norm(self.conv_0)
        #     self.conv_1 = spectral_norm(self.conv_1)
        #     if self.learned_shortcut:
        #         self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        self.norm_0 = SPADE(fin, lab_nc)
        self.norm_1 = SPADE(fmiddle, lab_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, lab_nc)

    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

# ==================================== mask SPADE =========================================
class maskSPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        ks = 3

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        # pw = ks // 2
        # self.mlp_shared = nn.Sequential(
            # nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            # nn.ReLU()
        # )
        # self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        # self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        
        self.mlp_shared = nn.Sequential(
            nn.Linear(256, nhidden),
            nn.ReLU()
        )
        
        self.mlp_gamma = nn.Linear(nhidden, norm_nc)
        self.mlp_beta = nn.Linear(nhidden, norm_nc)

    def forward(self, x, noise, mask):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        mask = F.interpolate(mask, size=x.size()[2:], mode='nearest')
        
        actv = self.mlp_shared(noise)
        gamma = self.mlp_gamma(actv).unsqueeze(-1).unsqueeze(-1).expand_as(x) * mask
        beta = self.mlp_beta(actv).unsqueeze(-1).unsqueeze(-1).expand_as(x) * mask

        # apply scale and bias
        out = normalized * (1. + gamma) + beta

        return out

class maskSPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, lab_nc, dilation=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        padding = dilation
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=padding, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=padding, dilation=dilation)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # define normalization layers
        self.norm_0 = maskSPADE(fin, lab_nc)
        self.norm_1 = maskSPADE(fmiddle, lab_nc)
        if self.learned_shortcut:
            self.norm_s = maskSPADE(fin, lab_nc)

    def forward(self, x, n, mask):
        x_s = self.shortcut(x, n, mask)

        dx = self.conv_0(self.actvn(self.norm_0(x, n, mask)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, n, mask)))

        out = x_s + dx

        return out

    def shortcut(self, x, n, mask):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, n, mask))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

# ==============================================================================================================

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim2=None, activation=nn.LeakyReLU(0.2), kernel_size=3, dilation=1):
        super().__init__()

        
        if dim2 == None:
            dim2 = dim
            # self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
            self.shortcut = lambda x: x
        else:
            self.shortcut = nn.Conv2d(dim, dim2, kernel_size=1, bias=False)

        pw = (kernel_size - 1) // 2 + (dilation-1)
        self.conv_block = nn.Sequential(
            nn.ZeroPad2d(pw),
            nn.InstanceNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, dilation=dilation),
            activation,
            nn.ZeroPad2d(pw),
            nn.InstanceNorm2d(dim),
            nn.Conv2d(dim, dim2, kernel_size=kernel_size, dilation=dilation)
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = self.shortcut(x) + y
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, dilation=1, norm='in', activation='lrelu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()

        self.use_bias = False
        if norm == 'in':
            self.use_bias = True

        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = spectral_norm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding=0, dilation=dilation, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding=0, dilation=dilation, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class UpConv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='in', activation='lrelu', pad_type='zero', up_mode='nearest'):
        super(UpConv2dBlock, self).__init__()

        self.use_bias = False
        if norm == 'IN':
            self.use_bias = True

        self.up = nn.Upsample(scale_factor=2, mode=up_mode)

        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = spectral_norm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat((x, skip), dim=1)
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='lrelu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

    
# ====================================  styleSPADE =========================================

class clsSPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        ks = 3

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, cls_map):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        cls_map = F.interpolate(cls_map, size=x.size()[2:], mode='nearest')

        actv = self.mlp_shared(cls_map)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1. + gamma) + beta

        return out


class styleSPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        ks = 3

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, sty_code):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        bs,cn,ww,hh = list(sty_code.size()[:2]) + list(x.size()[2:])
        sty_map = sty_code.expand(bs,cn,ww,hh)

        actv = self.mlp_shared(sty_map)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1. + gamma) + beta

        return out


class styleSPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, ins_nc, sty_nc, dilation=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        padding = dilation
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=padding, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=padding, dilation=dilation)
        # if self.learned_shortcut:
        #     self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # define normalization layers
        self.norm_0 = clsSPADE(fin, ins_nc)
        self.norm_1 = styleSPADE(fmiddle, sty_nc)

    def forward(self, x, cls_map, sty_code):
        x_s = x

        dx = self.conv_0(self.actvn(self.norm_0(x, cls_map)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, sty_code)))

        out = x_s + dx

        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
    

# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='reflect',
                 activation='elu', norm='none', sn=False):
        super(GatedConv2d, self).__init__()
        
        self.out_channels = out_channels
        
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
            self.mask_conv2d = spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
            self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.pad(x)
        conv = self.conv2d(x)
        if self.out_channels == 3:
            return conv
        
        if self.norm:
            conv = self.norm(conv)
        if self.activation:
            conv = self.activation(conv)        
        
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)
        x = conv * gated_mask

        return x


class TransposeGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero',
                 activation='lrelu', norm='none', sn=True, scale_factor=2):
        super(TransposeGatedConv2d, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type,
                                        activation, norm, sn)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        if skip is not None:
            x = torch.cat((x, skip), dim=1)
        x = self.gated_conv2d(x)
        return x

####################################################################################################################
####################################################################################################################
class CondGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, label_nc, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero',
                 activation='elu', norm='none', sn=False):
        super(CondGatedConv2d, self).__init__()
        
        self.out_channels = out_channels
        
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
            # self.mask_conv2d = spectral_norm(
                # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
            # self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
        self.sigmoid = torch.nn.Sigmoid()
        
        ####### mod 1 ########
        # nhidden = out_channels // 2
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(in_channels, nhidden, kernel_size=3, stride=stride, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, out_channels, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, out_channels, kernel_size=3, padding=1)

        ####### mod 2 ########
        self.mlp_shared_2 = nn.Sequential(
            nn.Conv2d(label_nc+1, nhidden, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma_ctx_gamma = nn.Conv2d(nhidden, out_channels, kernel_size=3, padding=1)
        self.mlp_beta_ctx_gamma = nn.Conv2d(nhidden, out_channels, kernel_size=3, padding=1)

        self.mlp_gamma_ctx_beta = nn.Conv2d(nhidden, out_channels, kernel_size=3, padding=1)
        self.mlp_beta_ctx_beta = nn.Conv2d(nhidden, out_channels, kernel_size=3, padding=1)

    def forward(self, x, seg, mask):
        x_pad = self.pad(x)
        conv = self.conv2d(x_pad)
        if self.out_channels == 3:
            return conv
        
        if self.norm:
            normalized = self.norm(conv)

        ####### mod 2 ########
        seg = F.interpolate(seg, size=normalized.size()[2:], mode='nearest')
        mask = F.interpolate(mask, size=normalized.size()[2:], mode='nearest')
        ctx = self.mlp_shared_2(torch.cat((seg, mask), dim=1))
        gamma_ctx_gamma = self.mlp_gamma_ctx_gamma(ctx)
        beta_ctx_gamma = self.mlp_beta_ctx_gamma(ctx)
        gamma_ctx_beta = self.mlp_gamma_ctx_beta(ctx)
        beta_ctx_beta = self.mlp_beta_ctx_beta(ctx)

        ####### mod 1 ########
        actv = self.mlp_shared(x)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        
        # print(gamma_ctx_gamma.size())
        # print(beta_ctx_gamma.size())
        # print(gamma.size())
        
        gamma = gamma * (1. + gamma_ctx_gamma) + beta_ctx_gamma
        beta = beta * (1. + gamma_ctx_beta) + beta_ctx_beta
        out_norm = normalized * (1. + gamma) + beta
        
        if self.activation:
            out = self.activation(out_norm)
        
        return out


class CondTransposeGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, label_nc, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero',
                 activation='lrelu', norm='none', sn=True, scale_factor=2):
        super(CondTransposeGatedConv2d, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = CondGatedConv2d(in_channels, out_channels, label_nc, kernel_size, stride, padding, dilation, pad_type,
                                        activation, norm, sn)

    def forward(self, x, seg, mask, skip=None):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        if skip is not None:
            x = torch.cat((x, skip), dim=1)
        x = self.gated_conv2d(x, seg, mask)
        return x

