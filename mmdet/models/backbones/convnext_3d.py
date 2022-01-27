# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import trunc_normal_init, build_conv_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.runner import BaseModule
from ..builder import BACKBONES
from functools import partial
from ...utils import get_root_logger, print_tensor
import ipdb

# from timm.models.layers import trunc_normal_, DropPath
# from timm.models.registry import register_model

class CNextBlock3D(BaseModule):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, expand_ratio = 4, 
                dropout_layer=dict(type='DropPath', drop_prob=0.), 
                layer_scale_init_value=1e-6, 
                dw_kernel_size = 5, 
                init_cfg = None):
        super(CNextBlock3D, self).__init__(init_cfg)
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=dw_kernel_size, 
                                padding= (dw_kernel_size - 1) // 2 , groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expand_ratio * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expand_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        
        # print('[CNextBlock] dwconv kernel', self.dwconv)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = build_dropout(dropout_layer)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W, D) -> (N, H, W, D, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3) # (N, H, W, D, C) -> (N, C, H, W, D)

        x = input + self.drop_path(x)
        return x

@BACKBONES.register_module()
class ConvNeXt3D(BaseModule):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """


    def __init__(self, 
                in_channels=1, 
                stem_cfg = dict(conv1kernel = 3, conv1stride = 1, conv1_chn_div = 2, 
                                conv2kernel = 3, conv2stride = 1), 
                expand_ratio = 4, 
                dw_kernel_size = 5, 
                depths=[0, 3, 3, 9, 3], 
                dims=[24, 48, 96, 192, 384], 
                drop_path_rate=0., 
                layer_scale_init_value=1e-6, 
                out_indices=(0, 1, 2, 3, 4),
                frozen_stages=-1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', requires_grad=True),
                init_cfg = [dict(type='Kaiming', layer=['Conv3d', 'Linear']),
                            dict(type='Constant', val = 1, layer = 'LayerNorm')
                            ], 
                **kwargs
                ):
        super(ConvNeXt3D, self).__init__(init_cfg=init_cfg)

        self.stem_cfg = stem_cfg
        self.in_channels = in_channels
        self.num_features = dims
        self.out_indices = out_indices
        self.num_stages = len(depths)
        self.strides = [stem_cfg.get('conv1stride', 1) * stem_cfg.get('conv2stride', 1)] \
                        + [2] * (self.num_stages - 1)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dw_kernel_size = dw_kernel_size

        self.stem_layer = self._make_stem_layer(in_channels, dims[0], stem_cfg)

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        # stem = nn.Sequential(
        #     nn.Conv3d(in_channels, dims[0], kernel_size=stem_cfg['kernel_size'], stride=stem_cfg['stride']),
        #     LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        # )
        # self.downsample_layers.append(stem)
    
        for i in range(self.num_stages - 1):
            downsample_layer = nn.Sequential(
                nn.Identity() if i == 0 else LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(self.num_stages - 1):
            stage = nn.Sequential(
                *[CNextBlock3D(dim=dims[i+1], 
                            expand_ratio = expand_ratio, 
                            dropout_layer=dict(type='DropPath', drop_prob=dp_rates[cur + j]),  #drop_path=dp_rates[cur + j], 
                            layer_scale_init_value=layer_scale_init_value, 
                            dw_kernel_size = dw_kernel_size, 
                ) for j in range(depths[i+1])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)
        # self.apply(self._init_weights)

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(self.num_stages - 1):
            layer = norm_layer(dims[i_layer+1])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def _make_stem_layer(self, in_channels, stem_channels, 
                    stem_cfg = dict(conv1kernel = 3, conv1stride = 1, conv1_chn_div = 2, 
                                    conv2kernel = 3, conv2stride = 1)):
        """Make stem layer for ResNet.
        
        i = input size, o = output size, p = padding, k = kernel_size, s = stride, d = dilation
        o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
        if i == o: then (o - 1)s = (i - 1)*s = i + 2*p - k - (k-1)*(d-1)
        p = (k + (k - 1)*(d-1) + i*(s -1) - s)/2
        if d = 1, p = (k + i *(s -1) -s)/2

        """
        stem_channel_div = stem_cfg.get('conv1_chn_div', 1)
        conv1kernel = stem_cfg.get('conv1kernel', 3)
        conv1stride = stem_cfg.get('conv1stride', 1)
        conv1pad = (conv1kernel - 1 )//2 #if (conv1kernel % conv1stride != 0) else 0
        conv2kernel = stem_cfg.get('conv2kernel', 3)
        conv2stride = stem_cfg.get('conv2stride', 1)
        conv2pad = (conv2kernel - 1 )//2
        
        stem_layer = nn.Sequential(
            build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels // stem_channel_div, # 
                kernel_size=[conv1kernel] *3,
                stride=[conv1stride] * 3,
                padding=[conv1pad] * 3,
                bias=False),
            LayerNorm(stem_channels // stem_channel_div, eps=1e-6, data_format="channels_first"), 
            # build_norm_layer(self.norm_cfg, stem_channels)[1], #// 2
            nn.GELU(),
            build_conv_layer(
                self.conv_cfg,
                stem_channels // stem_channel_div,#
                stem_channels ,#// 2
                kernel_size=[conv2kernel] *3,
                stride= [conv2stride] * 3,
                padding=[conv2pad] * 3,
                bias=False),
            LayerNorm(stem_channels, eps=1e-6, data_format="channels_first"), 
            # build_norm_layer(self.norm_cfg, stem_channels)[1],#// 2
            nn.GELU(),
            )
        return stem_layer

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x = self.forward_features(x)
        # x = self.head(x)
        x = self.stem_layer(x)
        outs = [x]
        for i in range(self.num_stages - 1 ):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            norm_layer = getattr(self, f'norm{i}')
            x_out = norm_layer(x)
            # print_tensor(f'[ConvNext] level i {i}', x_out)
            outs.append(x_out)

        return tuple([outs[i] for i in self.out_indices])



@BACKBONES.register_module()
class ConvNeXt3D4SimMIM(ConvNeXt3D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # assert self.num_classes == 0
        # self.input_size = input_size
        # self.output_size = [s//2**5 for i, s in enumerate(input_size)]
        self.mask_token = nn.Parameter(torch.zeros(1, self.num_features[0]))

        trunc_normal_init(self.mask_token, mean=0., std=.02)
        # self.patch_size = 0

    def forward(self, x, mask):
        x = self.stem_layer(x)
        assert mask is not None
        # B, L, H, W, D = x.shape
        # not masking the original image, but masking the embedding features !! 
        # Also, create learnable parameters to weight the masked region 
        mask_tokens = self.mask_token[..., None, None, None]
        ww = mask.unsqueeze(1).to(mask_tokens.dtype) # B, L, 1
        x = x * (1. - ww) + mask_tokens * ww
        # if self.use_abs_pos_embed:
        #     x = x + self.absolute_pos_embed
        # x = self.drop_after_pos(x)
        outs  = [x]
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            norm_layer = getattr(self, f'norm{i}')
            x_out = norm_layer(x)
            # print_tensor(f'[ConvNext] level {i} out', x_out) 
            outs.append(x_out)

        return tuple([outs[i] for i in self.out_indices])

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}



class LayerNorm(BaseModule):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last",
                init_cfg = None):
        super(LayerNorm, self).__init__(init_cfg)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # ipdb.set_trace()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

# @BACKBONES.register_module()
def convnext_tiny(pretrained=False, **kwargs):
    model = ConvNeXt3D(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

# @BACKBONES.register_module()
def convnext_small(pretrained=False, **kwargs):
    model = ConvNeXt3D(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

# @BACKBONES.register_module()
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt3D(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

# @BACKBONES.register_module()
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt3D(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

# @BACKBONES.register_module()
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt3D(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        url = model_urls['convnext_xlarge_22k'] if in_22k else model_urls['convnext_xlarge_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model