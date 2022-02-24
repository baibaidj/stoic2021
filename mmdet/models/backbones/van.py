
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.runner import BaseModule
from mmcv import Timer
from ..builder import BACKBONES
from functools import partial
from ...utils import get_root_logger, print_tensor
import ipdb

# from timm.models.layers import trunc_normal_, DropPath



@BACKBONES.register_module()
class VAN3D(BaseModule):
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
                mlp_ratio = 3, 
                lka_cfg = dict(dw_kernel_size = 5, 
                                dwd_kernel_size = 7, 
                                dwd_dilation = 3), 
                depths=[3, 3, 9, 3], 
                dims=[24, 48, 96, 192], 
                drop_path_rate=0., 
                layer_scale_init_value=1e-6, 
                out_indices=(0, 1, 2, 3),
                frozen_stages=-1,
                conv_cfg=dict(type = 'Conv3d'),
                norm_cfg=dict(type='BN3d', requires_grad=True) ,  
                init_cfg = [dict(type='TruncNormal', std = 0.2, layer=['Conv3d', 'Linear']), #Xavier
                            dict(type='Constant', val = 1, layer = ['LayerNorm'])
                            ], 
                verb = False,
                **kwargs
                ):
        super(VAN3D, self).__init__(init_cfg=init_cfg)

        # self.stem_cfg = stem_cfg
        self.in_channels = in_channels
        self.num_features = dims
        self.out_indices = out_indices
        self.num_stages = len(depths)
        self.strides = [4] + [2] * (self.num_stages - 1)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.lka_cfg = lka_cfg
        self.frozen_stages = frozen_stages
        self.fp16_enabled = False
        self.verb = verb

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers

        for i in range(self.num_stages):
            downsample_layer = OverlapPatchEmbed(in_channels=in_channels if i == 0 else dims[i - 1], 
                                                out_channels=dims[i], 
                                                kernel_size= 7 if i == 0 else 3, 
                                                stride = 4 if i == 0 else 2, 
                                                conv_cfg = conv_cfg, 
                                                norm_cfg = dict(type = 'LN', requires_grad=True)
                                                )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(self.num_stages):
            stage = nn.Sequential(
                *[VANBlock(dim=dims[i], 
                            mlp_ratio = mlp_ratio, 
                            dropout_layer=dict(type='DropPath', drop_prob=dp_rates[cur + j]),  #drop_path=dp_rates[cur + j], 
                            layer_scale_init_value=layer_scale_init_value, 
                            lka_cfg = lka_cfg, 
                            norm_cfg= norm_cfg
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]


        norm_layer = partial(NormLayer, eps=1e-6, data_format="channels_first")
        for i_layer in range(self.num_stages):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

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

        outs = []
        for i in range(self.num_stages):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            norm_layer = getattr(self, f'norm{i}')
            x_out = norm_layer(x)
            if self.verb: print_tensor(f'[VAN] level i {i}', x_out)
            outs.append(x_out)

        return tuple([outs[i] for i in self.out_indices])


class VANBlock(BaseModule):
    def __init__(self, 
                dim, 
                mlp_ratio=4., 
                conv_cfg = None, 
                norm_cfg = dict(type='BN3d', requires_grad=True), 
                lka_cfg = dict(dw_kernel_size = 5, 
                                dwd_kernel_size = 7, 
                                dwd_dilation = 3), 
                layer_scale_init_value=1e-2,
                dropout_layer=dict(type='DropPath', drop_prob=0.), 
                init_cfg = None, 
                ):
        super(VANBlock, self).__init__(init_cfg = init_cfg)

        norm1_name, self.norm1 = build_norm_layer(norm_cfg, dim)
        self.attn = SpatialAttention(dim, **lka_cfg)
        self.drop_path = build_dropout(dropout_layer)

        norm2_name, self.norm2 = build_norm_layer(norm_cfg, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, dropout_layer=dropout_layer)
        layer_scale_init_value = 1e-2      

        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x_norm1 = self.norm1(x)
        # with Timer(print_tmpl='\t VAN att {:.3f} seconds'):
        x_attn = self.attn(x_norm1)
        x = x + self.drop_path(self.layer_scale_1[None, :, None, None, None] * x_attn)

        x_norm2 = self.norm2(x)
        # with Timer(print_tmpl='\t VAN mlp {:.3f} seconds'):
        x_mlp = self.mlp(x_norm2)
        x = x + self.drop_path(self.layer_scale_2[None, :, None, None, None] * x_mlp)
        return x

class SpatialAttention(BaseModule):
    def __init__(self, dim, 
                dw_kernel_size = 5, 
                dwd_kernel_size = 5, 
                dwd_dilation = 3,
                init_cfg = dict(type='TruncNormal', std = 0.2, layer='Conv3d')):
        super(SpatialAttention, self).__init__(init_cfg=init_cfg)

        self.proj_1 = nn.Conv3d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(dim, 
                                            dw_kernel_size,
                                            dwd_kernel_size, 
                                            dwd_dilation)
        self.proj_2 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        shorcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        # with Timer(print_tmpl='\t InnerAtt {:.3f} seconds'):
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class AttentionModule(BaseModule):
    """
    i = input size, o = output size, p = padding, k = kernel_size, s = stride, d = dilation
    o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
    if i == o: 
        then (o - 1)s = (i - 1)*s = i + 2*p - k - (k-1)*(d-1)

    p = (k + (k - 1)*(d-1) + i*(s -1) - s)/2
    if d = 1, p = (k + i *(s -1) -s)/2

    stride = 1
    p = (k + (k - 1)*(d-1) + i*(s -1) - s)/2
    p = (k + (k -1) * (d - 1) - 1)/2

    """

    def __init__(self, dim, 
                dw_kernel_size = 5, 
                dwd_kernel_size = 5, 
                dwd_dilation = 3,
                init_cfg = dict(type='TruncNormal', std = 0.2, layer='Conv3d')):
        super(AttentionModule, self).__init__(init_cfg = init_cfg)

        pad_fun = lambda k, d :  (k + (k -1) * (d - 1) - 1)//2
        pad4dw = pad_fun(dw_kernel_size, 1)
        self.conv0 = nn.Conv3d(dim, dim, dw_kernel_size, padding=pad4dw, groups=dim)

        pad4dwd= pad_fun(dwd_kernel_size, dwd_dilation)
        self.conv_spatial = nn.Conv3d(dim, dim, 
                                     dwd_kernel_size, 
                                     dilation=dwd_dilation, 
                                     padding=pad4dwd, 
                                     groups=dim, )

        self.conv1 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        u = x
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        out = u * attn

        # print_tensor('[LKA] in', u)
        # print_tensor('[LKA] attn', attn)
        # print_tensor('[LKA] out', out)
        return out


class Mlp(BaseModule):
    def __init__(self, in_features, 
                hidden_features=None, 
                out_features=None, 
                dropout_layer=dict(type='DropPath', drop_prob=0.), 
                init_cfg = None):
        super(Mlp, self).__init__(init_cfg = init_cfg)

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv3d(hidden_features, hidden_features, 
                                3, 1, 1, bias=True, groups=hidden_features) #DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = build_dropout(dropout_layer)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class OverlapPatchEmbed(BaseModule):
    """ Image to Patch Embedding
    """

    def __init__(self, 
                    in_channels=1, 
                    out_channels=32, 
                    kernel_size=7, 
                    stride=4, 
                    conv_cfg = None, 
                    norm_cfg = None,
                    init_cfg = None):
        super(OverlapPatchEmbed, self).__init__(init_cfg = init_cfg)

        self.patch_embed = nn.Sequential(
                            build_conv_layer(
                                conv_cfg,
                                in_channels,
                                out_channels, # 
                                kernel_size=[kernel_size] * 3,
                                padding= (kernel_size - 1 )//2, 
                                stride=[stride] * 3,
                                bias=False),
                            NormLayer(out_channels, eps=1e-6, 
                                    data_format="channels_first", 
                                    norm_cfg=norm_cfg)
            )
    def forward(self, x):
        x = self.patch_embed(x)     
        return x

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


class NormLayer(BaseModule):

    def __init__(self, normalized_shape,
                data_format="channels_last", eps=1e-6, 
                norm_cfg = dict(type = 'LN'), init_cfg=None):
        super().__init__(init_cfg)

        self.use_ln = norm_cfg['type'] == 'LN'
        if self.use_ln:
            self.norm_layer = LayerNorm(normalized_shape, eps = eps, 
                                        data_format=data_format)
        else:
            name, self.norm_layer = build_norm_layer(norm_cfg, normalized_shape)
    
    def forward(self, x):
        # with Timer(print_tmpl='\t Norm {:.3f} seconds' + '%s'%(self.norm_layer)):
        x = self.norm_layer(x)
        return x