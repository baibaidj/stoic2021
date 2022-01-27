from torch.nn.modules import conv
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nl4skip import NonLocalGeneric, NL4Skip_CR, print_tensor

from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer, build_upsample_layer
from copy import deepcopy


from ..backbones.resnet_3d_iso import BasicBlock, BasicBlockP3D, Bottleneck, BottleneckP3D
# from .up_conv_block import SemanticFlowUpsample3D
from .ccnet_pure import CCAttention3D
import pdb

class DecoderBlock_Vnet(nn.Module):

    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            is_nlblock = False,
            level_ix = 0,
            use_sece=None,
            deconv_cfg = dict(type='deconv3d', kernel_size = (2,2,2), stride = (2,2,2)),
            conv_cfg = dict(type = 'Conv2d'),
            norm_cfg = None,
            act_cfg = dict(type='ReLU', inplace = True),
            align_corners = False,
            is_p3d = False,
            is_bottleneck = False, 
            upsample_type = 'deconv',

            # is_nlblock = False,
            # nl_fusion_type = 'add',
            # nl_psp_size = (1,3,6,8),
            # nl_key_source = 'self',
            # cc_post_conv2 = False, 
            # cc_reduction = 8,
            nl_cfg = dict(type = 'naive', 
                         fusion_type = 'add', 
                         key_source = 'self', 
                         psp_size = None, out_projection = True,
                         reduction = 8)
    ):

        # in_channels = [head_channels] + list(decoder_channels[:-1]) # 2048 256 128 64 32
        # skip_channels = list(encoder_channels[1:]) + [0] # 1024, 512, 256, 64, 0

        super(DecoderBlock_Vnet, self).__init__()
        assert upsample_type in ('deconv', 'semanticflow')
        # using transpose conv to upsample earlier feat x and then cat it with the skip
        #  reduce channels of x, 
        # assume that in_channels; skip_channels = out_channels / 2

        self.cc_post_conv2 = bool(is_nlblock) & (nl_cfg.get('type', 'naive') == 'cca')
        self.is_nlblock = bool(is_nlblock)&(skip_channels>0) & (not self.cc_post_conv2)
        self.nl_fusion_type = nl_cfg.get('fusion_type', 'add')
        self.nl_key_source = nl_cfg.get('key_source', 'self')
        self.nl_psp_size = nl_cfg.get('psp_size', None)
        self.nl_reduction = nl_cfg.get('reduction', 8)
        self.nl_out_projection = nl_cfg.get('out_projection', True)
        self.align_corners = align_corners

        up_channels = out_channels //2 #max(skip_channels , 16)
        self.is_3d = False if conv_cfg is None else (True if '3d' in conv_cfg.get('type', '').lower() else False)
        print('decoder: ip-%04d\tskip-%04d\tisnl-%s\tatt-%s\tfuse%s cc%s' %(
                in_channels, skip_channels, self.is_nlblock, use_sece, self.nl_fusion_type, self.cc_post_conv2))

        self.upsample_type = upsample_type
        if self.upsample_type == 'deconv':
            self.upsample_layer = nn.Sequential(
                                        build_upsample_layer(cfg=deconv_cfg,
                                                in_channels=in_channels,
                                                out_channels=up_channels,
                                                bias = False),
                                        build_norm_layer(norm_cfg, up_channels)[1],
                                        nn.ReLU(inplace=True)
                                        )
        # else:
        #     self.upsample_layer = SemanticFlowUpsample3D(in_channels, skip_channels, conv_cfg= conv_cfg) 
        #     self.reduce_flow_dim = ConvModule(in_channels, skip_channels, 1, conv_cfg= conv_cfg, 
        #                                      norm_cfg=norm_cfg)


        if self.is_nlblock:
            print('NL-HLL: key=value')
            self.NLskip = NL4Skip_CR(skip_channels, skip_channels, sub_sample=False, psp_size = self.nl_psp_size,
                                    dimension= 2 + int(self.is_3d))

        conv2ip_ch = skip_channels + up_channels
        if self.is_nlblock and self.nl_fusion_type=='add': conv2ip_ch = skip_channels

        print('\tconv2ip_ch %s ' %(conv2ip_ch))
        
        conv_kwargs2 = dict(inplanes = conv2ip_ch,
                           planes = out_channels,
                          # kernel_size=3,
                          # padding=1,
                          conv_cfg = conv_cfg,  
                          norm_cfg = norm_cfg,
                          downsample = (None if conv2ip_ch == out_channels else 
                                        ConvModule(conv2ip_ch, out_channels, 1,
                                        conv_cfg= conv_cfg, norm_cfg=norm_cfg, act_cfg=None))
                          )
        if is_p3d:
            ConvBlock = BottleneckP3D if is_bottleneck else BasicBlockP3D
            # conv_kwargs1['global_ix'] = level_ix * 2
            conv_kwargs2['global_ix'] = level_ix * 1 + 1
            if not is_bottleneck: conv_kwargs2['is_double'] = False
        else:
            ConvBlock = Bottleneck if is_bottleneck else BasicBlock

        self.conv2 = ConvBlock(**conv_kwargs2)
        # self.attention1 = Attention(use_sece, is_3d = self.is_3d, in_channels=up_channels)
        if self.cc_post_conv2:
            self.attention2 = CCAttention3D(out_channels, conv_cfg = conv_cfg, 
                                            norm_cfg= norm_cfg, 
                                            act_cfg= act_cfg, 
                                            out_projection= self.nl_out_projection,
                                            reduction=self.nl_reduction) 
        else: self.attention2 = AttentionSECE(use_sece, is_3d = self.is_3d, in_channels=out_channels)

    def forward(self, x, skip):
        """
        x is of lower resolution, skip is of higher
        """
        # get_shape = lambda x: x.shape[-3:] if self.is_3d else x.shape[-2:]
        # get_mode = lambda : 'trilinear' if self.is_3d else 'bilinear'
        
        if self.upsample_type == 'deconv':
            x = self.upsample_layer(x)
        else:
            h_flow = self.upsample_layer(x, skip) # 
            h_flow = self.reduce_flow_dim(h_flow)
            x = h_flow
            # x = x + skip_reduce

        # x = self.attention1(x)
        if self.is_nlblock:
            # print('NL for ', x.shape, skip.shape, )
            context = self.NLskip(x, skip)
            if self.nl_fusion_type == 'add':
                x = context + x
            else:
                x = torch.cat([x, context], dim=1)
        else:
            # print(x.shape, skip.shape)
            x = torch.cat([x, skip], dim=1)
        # print_tensor('\tBefore SE', x)
        # pdb.set_trace()
        x = self.conv2(x)
        # print_tensor('\n\ndecoder, concat', x)
        x = self.attention2(x)
        # print_tensor('After SE', x)
        return x



class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, conv_cfg, norm_cfg):
        
        # self.add_module(self.norm1_name, norm1)
        middle_channel = int(in_channels * 2)
        relu1 = relu(True)
        conv1 = ConvModule(in_channels, middle_channel,
                            3, 
                            padding = 1,
                            conv_cfg=conv_cfg,
                            norm_cfg = None
                            )
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, middle_channel)

        conv2 = ConvModule(middle_channel, in_channels,
                            3, 
                            padding = 1,
                            conv_cfg= conv_cfg,
                            norm_cfg = None
                            )
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, in_channels)

        super(CenterBlock, self).__init__(conv1, norm1, relu1, conv2, norm2, relu1)



def relu(inplace:bool=False, leaky:float=None):
    "Return a relu activation, maybe `leaky` and `inplace`."
    return nn.LeakyReLU(inplace=inplace, negative_slope=leaky) if leaky is not None else nn.ReLU(inplace=inplace)

def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function."
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
    x.data.copy_(k)

def ifnone(a ,b):
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a


class PixelShuffle_ICNR(nn.Module):
    '''Upsample by `scale` from `ni` filters to `nf` (default `ni`), 
    using `nn.PixelShuffle`, `icnr` init, and `weight_norm`.
    the output will have nf number of channels.
    '''
    def __init__(self, ni:int, nf:int=None, scale:int=2, blur:bool=True, norm_cfg=None, leaky:float=None):
        super().__init__()
        nf = ifnone(nf, ni)
        self.conv = ConvModule(ni, nf*(scale**2), 1, act_cfg = None)
        # icnr(self.conv.weight)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1,0,1,0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.do_blur = blur
        self.relu = relu(True, leaky=leaky)

    def forward(self,x):
        x = self.shuf(self.relu(self.conv(x))) 
        return self.blur(self.pad(x)) if self.do_blur else x



class ArgMax(nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Activation(nn.Module):

    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)


class AttentionSECE(nn.Module):

    def __init__(self, name, is_3d = False, **params):
        super().__init__()

        if bool(name):
            self.attention = SCSEModule3D(**params) if is_3d else SCSEModule(**params)
        else:
            self.attention = nn.Identity(**params)
        # else:
        #     raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(in_channels // reduction, 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_channels // reduction, 4), in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class SCSEModule3D(nn.Module):
    def __init__(self, in_channels, reduction=8, scale = 1e-4, verbose = False):
        super().__init__()
        self.scale = scale
        self.verbose = verbose
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.cSE = nn.Sequential(
            nn.Conv3d(in_channels, max(in_channels // reduction, 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(in_channels // reduction, 4), in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv3d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        cse_ip = self.avg_pool(x * self.scale)/ self.scale
        if self.verbose: print_tensor('AVG pool', cse_ip)
        return x * self.cSE(cse_ip) + x * self.sSE(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)




# class DecoderBlock(nn.Module):
#     def __init__(
#             self,
#             in_channels,
#             skip_channels,
#             out_channels,
#             use_batchnorm=True,
#             attention_type=None,
#     ):
#         super().__init__()
#         self.conv1 = Conv2dReLU(
#             in_channels + skip_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
#         self.conv2 = Conv2dReLU(
#             out_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         self.attention2 = Attention(attention_type, in_channels=out_channels)

#     def forward(self, x, skip=None):
#         x = F.interpolate(x, scale_factor=2, mode="nearest")
#         if skip is not None:
#             x = torch.cat([x, skip], dim=1)
#             x = self.attention1(x)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.attention2(x)
#         return x
