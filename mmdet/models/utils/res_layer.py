# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.models.backbones.resnet3d import Bottleneck3d
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, Sequential
from torch import nn as nn
from torch.nn.modules.utils import _ntuple, _triple
import torch

class ResLayer(Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 downsample_first=True,
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))

        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)


class SimplifiedBasicBlock(BaseModule):
    """Simplified version of original basic residual block. This is used in
    `SCNet <https://arxiv.org/abs/2012.10150>`_.

    - Norm layer is now optional
    - Last ReLU in forward function is removed
    """
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_fg=None):
        super(SimplifiedBasicBlock, self).__init__(init_fg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        assert not with_cp, 'Not implemented yet.'
        self.with_norm = norm_cfg is not None
        with_bias = True if norm_cfg is None else False
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=with_bias)
        if self.with_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, planes, postfix=1)
            self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=with_bias)
        if self.with_norm:
            self.norm2_name, norm2 = build_norm_layer(
                norm_cfg, planes, postfix=2)
            self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name) if self.with_norm else None

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name) if self.with_norm else None

    def forward(self, x):
        """Forward function."""

        identity = x

        out = self.conv1(x)
        if self.with_norm:
            out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.with_norm:
            out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out


class ResLayer3D(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        multi_grid (int | None): Multi grid dilation rates of last
            stage. Default: None
        contract_dilation (bool): Whether contract first dilation of each layer
            Default: False
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 dilation=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 multi_grid=None,
                 contract_dilation=False,
                 **kwargs):
        
        self.block = block
        st_per_block = 2 if 'basic' in block.__name__.lower() else 1
        global_ix = kwargs.pop('global_ix', None)
        # print(block, dir(block), block.__name__.lower(), st_per_block)
        # print('res1', global_ix)
        downsample = None
        stride = [stride] * 3 if type(stride) is int else stride
        dilation = [dilation] * 3 if type(dilation) is int else dilation
        assert len(stride) == 3

        if max(stride) != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            # check dilation for dilated ResNet
            if avg_down and (max(stride) != 1 or max(dilation) != 1):
                conv_stride = 1
                downsample.append(
                    nn.AvgPool3d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if multi_grid is None:
            if max(dilation) > 1 and contract_dilation:
                first_dilation = [max(dilation[i] // 2, 1) for i in dilation] 
            else:
                first_dilation = dilation
        else:
            first_dilation = multi_grid[0]

        kwargs1 = dict(inplanes=inplanes,
                planes=planes,
                stride=stride,
                dilation=first_dilation,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                **kwargs)
        if global_ix is not None: kwargs1['global_ix'] = global_ix

        layers.append(block(**kwargs1))

        global_ix = global_ix + 1 * st_per_block if global_ix is not None else None
        # print('res2', global_ix)
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            kwargs_loop = dict(inplanes=inplanes,
                                planes=planes,
                                stride=1,
                                dilation=dilation if multi_grid is None else multi_grid[i],
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg,
                                **kwargs)
            if global_ix is not None: kwargs_loop['global_ix'] = global_ix

            layers.append(block(**kwargs_loop))

            global_ix = global_ix + 1 * st_per_block if global_ix is not None else None
            # print('resl', global_ix)
        super(ResLayer3D, self).__init__(*layers)


class ResLayerIso(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        multi_grid (int | None): Multi grid dilation rates of last
            stage. Default: None
        contract_dilation (bool): Whether contract first dilation of each layer
            Default: False
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 dilation=1,
                 avg_down=False,
                 conv_cfg=None,
                 non_local=0,
                 non_local_cfg=dict(),
                 norm_cfg=dict(type='BN'),
                 multi_grid=None,
                 contract_dilation=False,
                 **kwargs):
        self.block = block
        st_per_block = 2 if 'basic' in block.__name__.lower() else 1
        global_ix = kwargs.pop('global_ix', None)
        non_local = _ntuple(num_blocks)(non_local)
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1: 
                conv_stride = 1
                downsample.append(
                    nn.AvgPool3d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if multi_grid is None:
            if dilation > 1 and contract_dilation:
                first_dilation = dilation // 2
            else:
                first_dilation = dilation
        else:
            first_dilation = multi_grid[0]

        kwargs1 = dict(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                dilation=first_dilation,
                non_local=non_local[0] == 1,
                non_local_cfg=non_local_cfg,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                **kwargs)
        if global_ix is not None: kwargs1['global_ix'] = global_ix
        layers.append(block(**kwargs1))

        global_ix = global_ix + 1 * st_per_block if global_ix is not None else None
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            kwargs_loop = dict(inplanes=inplanes,
                                planes=planes,
                                stride=1,
                                dilation=dilation if multi_grid is None else multi_grid[i],
                                non_local=(non_local[i] == 1),
                                non_local_cfg=non_local_cfg,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg,
                                **kwargs)
            if global_ix is not None: kwargs_loop['global_ix'] = global_ix
            layers.append(block(**kwargs_loop))
            global_ix = global_ix + 1 * st_per_block if global_ix is not None else None

        super(ResLayerIso, self).__init__(*layers)


class CrossStageLayerIso(nn.Module):
    """ResLayer to build ResNet style backbone.
    128

    [ 1x1, 64  ]
    [ 3x3, 64  ] x 3
    [ 1x1, 256 ] 

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        multi_grid (int | None): Multi grid dilation rates of last
            stage. Default: None
        contract_dilation (bool): Whether contract first dilation of each layer
            Default: False
    """

    def __init__(self,
                 block : Bottleneck3d,
                 inplanes, # 128
                 planes, # 64
                 num_blocks,
                 stride=1,
                 dilation=1,
                 avg_down=False,
                 conv_cfg=None,
                 non_local=0,
                 non_local_cfg=dict(),
                 norm_cfg=dict(type='BN'),
                 multi_grid=None,
                 contract_dilation=False,
                 **kwargs):
        
        super(CrossStageLayerIso, self).__init__() # *layers

        self.block = block
        st_per_block = 2 if 'basic' in block.__name__.lower() else 1
        global_ix = kwargs.pop('global_ix', None)
        non_local = _ntuple(num_blocks)(non_local)
        conv_downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            conv_downsample = []
            conv_stride = stride
            if avg_down and stride != 1: 
                conv_stride = 1
                conv_downsample.append(
                    nn.AvgPool3d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            conv_downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion, # TODO: if use expansion 
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            conv_downsample = nn.Sequential(*conv_downsample)

        self.conv_downsample = conv_downsample # down sample channels

        block_out_chs = planes * block.expansion # 256
        inplanes_side = block_out_chs // 2 # 64
        planes_side = planes // 2 # 32
        layers = []
        if multi_grid is None:
            if dilation > 1 and contract_dilation:
                first_dilation = dilation // 2
            else:
                first_dilation = dilation
        else:
            first_dilation = multi_grid[0]

        kwargs1 = dict(
                inplanes=inplanes_side,
                planes=planes_side,
                stride=stride,
                dilation=first_dilation,
                non_local=non_local[0] == 1,
                non_local_cfg=non_local_cfg,
                downsample=None,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                **kwargs)
        if global_ix is not None: kwargs1['global_ix'] = global_ix
        layers.append(block(**kwargs1))

        global_ix = global_ix + 1 * st_per_block if global_ix is not None else None
        inplanes_side = planes_side * block.expansion #  # 32 x 4   = 128
        for i in range(1, num_blocks):
            kwargs_loop = dict(inplanes=inplanes_side, #  32 x 4   = 128
                                planes=planes_side, # 32
                                stride=1,
                                dilation=dilation if multi_grid is None else multi_grid[i],
                                non_local=(non_local[i] == 1),
                                non_local_cfg=non_local_cfg,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg,
                                **kwargs)
            if global_ix is not None: kwargs_loop['global_ix'] = global_ix
            layers.append(block(**kwargs_loop))
            global_ix = global_ix + 1 * st_per_block if global_ix is not None else None

        self.res_block = nn.Sequential(*layers)
        
        self.conv_transition_b = ConvModule(inplanes_side, block_out_chs//2, kernel_size=1, 
                                            conv_cfg = conv_cfg, 
                                            norm_cfg = norm_cfg, 
                                            act_cfg=kwargs.get('act_cfg', None))
        self.conv_transition = ConvModule(block_out_chs//2, block_out_chs,  kernel_size=1, 
                                          conv_cfg = conv_cfg, 
                                          norm_cfg = norm_cfg, 
                                          act_cfg=kwargs.get('act_cfg', None))

    def forward(self, x):
        if self.conv_downsample is not None:
            x = self.conv_downsample(x)
        split = x.shape[1] // 2
        xs, xb = x[:, : split], x[:, split:]
        xb = self.res_block(xb)
        xb = self.conv_transition_b(xb).continuous()
        out = self.conv_transition(torch.cat([xs, xb]), dim = 1)
        return out


