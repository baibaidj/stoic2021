import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_plugin_layer,
                      constant_init, kaiming_init)
from mmcv.runner import load_checkpoint, BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm
from torch.nn.modules.utils import _ntuple

from ..builder import BACKBONES
from ..utils import  ResLayerIso
from .resnet import BasicBlock, Bottleneck
from .resnet3d import BasicBlock3d, Bottleneck3d
from ...utils import get_root_logger
import torch

print_tensor = lambda n, x: print(n, type(x), x.dtype, x.shape, x.min(), x.max())
class BasicBlockP3D(nn.Module):
    
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
                #  ST_struc = 'A'
                global_ix = 0,
                is_double = True
                 ) -> None:
        super(BasicBlockP3D, self).__init__()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation[0] if type(dilation) in (tuple, list) else dilation
        self.with_cp = with_cp
        self.is_double = is_double
        # print('global %d, dilate %s' % (global_ix, dilation))
        self.conv_p3d_1 = Pseudo3DConv(inplanes, planes, stride = stride, dilation= self.dilation, 
                                    norm_cfg=norm_cfg, global_ix= global_ix)
        if self.is_double: self.conv_p3d_2 = Pseudo3DConv(planes, planes, stride = 1, dilation= self.dilation, 
                                    norm_cfg=norm_cfg, global_ix= global_ix + 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            # print('in', x.shape)
            out = self.conv_p3d_1(x)
            # print('out1', out.shape)
            if self.is_double: out = self.conv_p3d_2(out)
            # print('out2', out.shape)
            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

class BottleneckP3D(nn.Module):
    """Bottleneck block for ResNet.

    If style is "pytorch", the stride-two layer is the 3x3 conv layer, if it is
    "caffe", the stride-two layer is the first 1x1 conv layer.
    """

    expansion = 4

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
                #  ST_struct = 'A',
                 global_ix = 0
                 ):
        super(BottleneckP3D, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation[0] if type(dilation) in (tuple, list) else dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None    

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.conv_p3d = Pseudo3DConv(planes, planes, stride = self.conv2_stride, dilation = self.dilation,
                                    norm_cfg= norm_cfg, global_ix = global_ix)
    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            # if self.with_plugins:
            #     out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv_p3d(out)
            # out = self.norm2(out)
            # out = self.relu(out)

            # if self.with_plugins:
            #     out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            # if self.with_plugins:
            #     out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out        
#  deeper, output all skips, enable dilation, pseudo3d

@BACKBONES.register_module()
class ResNet3dIso(BaseModule):
    """ResNet3D backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default" 3.
        stem_channels (int): Number of stem channels. Default: 64.
        base_channels (int): Number of base channels of res layer. Default: 64.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.

            - position (str, required): Position inside block to insert plugin,
            options: 'after_conv1', 'after_conv2', 'after_conv3'.

            - stages (tuple[bool], optional): Stages to apply plugin, length
            should be same as 'num_stages'
        multi_grid (Sequence[int]|None): Multi grid dilation rates of last
            stage. Default: None
        contract_dilation (bool): Whether contract first dilation of each layer
            Default: False
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmseg.models import ResNet
        >>> import torch
        >>> self = ResNet3D(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))

        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2, 2)), 
        34: (BasicBlock, (3, 4, 6, 3, 2)),
        50: (Bottleneck, (3, 4, 6, 3, 2)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
        '34bneck': (Bottleneck, (2, 2, 2, 2)), 
        '18basicp3d': (BasicBlockP3D, (2, 2, 2, 2)), 
        '34basicp3d': (BasicBlockP3D, (3, 4, 6, 3)), 
        '34bneckp3d': (BottleneckP3D, (2, 2, 2, 2)),
        '50bneckp3d': (BottleneckP3D, (3, 4, 6, 3)),
        '101bneckp3d': (BottleneckP3D, (3, 4, 23, 3)), 
        '183d': (BasicBlock3d, (2, 2, 2, 2, 2)), 
        '343d': (BasicBlock3d, (3, 4, 6, 3, 2)),
        '503d': (Bottleneck3d, (3, 4, 6, 3, 2)),
    }

    p3d_structs = ['A', 'B', 'C']
    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1, 
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False, False),
                 plugins=None,
                 multi_grid=None,
                 contract_dilation=False,
                 with_cp=False,
                 zero_init_residual=True,
                 stem_stride_1 = 2,
                 stem_stride_2 = 1, 
                 stem_channel_div = 1,
                 non_local=(0, 0, 0, 0, 0),
                 non_local_cfg=dict(),
                 verbose = False, 
                 init_cfg = None,
                 ):
        # print('[ResNet3dIso] init cfg', init_cfg)
        super(ResNet3dIso, self).__init__(init_cfg=init_cfg)
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        # self.kernel_size = kernel_size
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 5
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        # assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.multi_grid = multi_grid
        self.contract_dilation = contract_dilation
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.non_local_stages = _ntuple(num_stages)(non_local)
        self.non_local_cfg = non_local_cfg
        self.inplanes = stem_channels
        self.is_p3d = True if 'p3d' in str(depth) else False
        self.verbose = verbose
        self.fp16_enabled = False
        # self.is_stem_down = is_stem_down


        self._make_stem_layer(in_channels, stem_channels, stem_stride_1, stem_stride_2, 
                                stem_channel_div = stem_channel_div)

        self.res_layers = []    
        global_block_ix = 0
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            # multi grid is applied to last layer only
            stage_multi_grid = multi_grid if i == len(
                self.stage_blocks) - 1 else None
            planes = base_channels * 2**i

            keyargs = dict(block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                multi_grid=stage_multi_grid,
                contract_dilation=contract_dilation, 
                non_local=self.non_local_stages[i],
                non_local_cfg=self.non_local_cfg,
                )
            
            if self.is_p3d: keyargs['global_ix'] = global_block_ix
            # print('\n stage %d args %s' %(i, global_block_ix))
            res_layer = self.make_res_layer(**keyargs)

            global_block_ix += num_blocks * (2 if 'basic' in str(self.depth) else 1)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i+1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)

    def make_stage_plugins(self, plugins, stage_idx):
        """make plugins for ResNet 'stage_idx'th stage .

        Currently we support to insert 'context_block',
        'empirical_attention_block', 'nonlocal_block' into the backbone like
        ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be :
        >>> plugins=[
        ...     dict(cfg=dict(type='xxx', arg1='xxx'),
        ...          stages=(False, True, True, True),
        ...          position='after_conv2'),
        ...     dict(cfg=dict(type='yyy'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='1'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='2'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3')
        ... ]
        >>> self = ResNet(depth=18)
        >>> stage_plugins = self.make_stage_plugins(plugins, 0)
        >>> assert len(stage_plugins) == 3

        Suppose 'stage_idx=0', the structure of blocks in the stage would be:
            conv1-> conv2->conv3->yyy->zzz1->zzz2
        Suppose 'stage_idx=1', the structure of blocks in the stage would be:
            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""

        # print('ResLayer Kargs', kwargs)
        return ResLayerIso(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels, stem_stride_1 = 1, stem_stride_2 = 1, 
                        stem_channel_div = 1):
        """Make stem layer for ResNet."""
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // stem_channel_div, # 
                    kernel_size=(3, 3, 3),
                    stride=[stem_stride_1] * 3,
                    padding=(1, 1, 1),
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1], #// 2
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // stem_channel_div,#
                    stem_channels ,#// 2
                    kernel_size=(3, 3, 3),
                    stride= [stem_stride_2] * 3,
                    padding=(1, 1, 1),
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],#// 2
                nn.ReLU(inplace=True),
                # build_conv_layer(
                #     self.conv_cfg,
                #     stem_channels // 2,
                #     stem_channels,
                #     kernel_size=3,
                #     stride=1,
                #     padding=1,
                #     bias=False),
                # build_norm_layer(self.norm_cfg, stem_channels)[1],
                # nn.ReLU(inplace=True)
                )
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=(7, 7, 7),
                stride=[stem_stride_1] * 3,
                padding=(3, 3, 3),
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool3d(kernel_size=3, 
        #                             stride=(2 if is_z_down else 1, 2, 2), 
        #                             padding=1)


    def _freeze_stages(self):
        """Freeze stages param and norm stats."""
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    # def init_weights(self, pretrained=None):
    #     """Initialize the weights in backbone.

    #     Args:
    #         pretrained (str, optional): Path to pre-trained weights.
    #             Defaults to None.
    #     """
    #     if isinstance(pretrained, str):
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, strict=False, logger=logger)
    #     elif pretrained is None:
    #         for m in self.modules():
    #             if isinstance(m, nn.Conv3d):
    #                 kaiming_init(m)
    #             elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
    #                 constant_init(m, 1)

    #         if self.dcn is not None:
    #             for m in self.modules():
    #                 if isinstance(m, Bottleneck) and hasattr(
    #                         m, 'conv3_offset'):
    #                     constant_init(m.conv3_offset, 0)

    #         if self.zero_init_residual:
    #             for m in self.modules():
    #                 if isinstance(m, Bottleneck):
    #                     constant_init(m.norm3, 0)
    #                 elif isinstance(m, BasicBlock):
    #                     constant_init(m.norm2, 0)
    #     else:
    #         raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function.
           outs: [input, 1/2, 1/4, 1/8, 1/16, 1/32] 6 items
        """
        outs = [x]
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        outs.append(x)

        if self.verbose: print_tensor('l1', x)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            # print(i, res_layer)
            x = res_layer(x)
            if self.verbose: print_tensor(f'l{i+2}', x)
            outs.append(x)
        
        # print('\n')
        # aa = [print(f'[Backbone] level {i} has nan? ', torch.isnan(t).any()) for i, t in enumerate(outs)]
        return tuple([outs[i] for i in self.out_indices])

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet3dIso, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

class Pseudo3DConv(nn.Module):
    
    st_structs = ['A', 'B', 'C']

    def __init__(self, in_planes, out_planes, stride = 1, dilation = 1, norm_cfg = dict(type='GN'), 
                global_ix = 0):
        super(Pseudo3DConv, self).__init__()        
        self.in_planes = in_planes
        self.out_planes = out_planes
        if type(stride) is int: stride = [stride] * 3
        assert len(stride) == 3
        self.stride_spatial = stride[1:]
        self.stride_temporal = stride[0]

        self.dilation = dilation
        self.norm_cfg = norm_cfg
        assert isinstance(global_ix, int)

        self.combine_type = 'A' if max(stride) > 1 else self.st_structs[global_ix%3]
        
        # self.norm_1 = build_norm_layer(self.norm_cfg, planes)[1]
        self.conv1 = self.SpatialConv0()
        self.norm1 = build_norm_layer(self.norm_cfg, out_planes)[1]

        self.conv2 = self.TemproalConv0() if self.combine_type == 'B' else self.TemproalConv1()
        self.norm2 = build_norm_layer(self.norm_cfg, out_planes)[1]
        self.relu = nn.ReLU(inplace=True)

        # self.
    # @property
    def SpatialConv0(self):
    # as is descriped, conv S is 1x3x3
        return nn.Conv3d(self.in_planes, self.out_planes, 
                        kernel_size=(1,3,3), 
                        stride=(1, self.stride_spatial[0], self.stride_spatial[1]),
                        padding= (0, self.dilation, self.dilation),
                        dilation = (1, self.dilation, self.dilation), 
                        bias=False)

    # @property
    def TemproalConv0(self,  is_temproal_down = True):
        # conv T is 3x1x1
        return nn.Conv3d(self.in_planes, self.out_planes, 
                        kernel_size=(3, 1, 1), 
                        stride= (self.stride_temporal if is_temproal_down else 1, 1, 1),
                        padding=(self.dilation, 0, 0), 
                        dilation = (self.dilation, 1, 1), bias=False)
    
    # @property
    def TemproalConv1(self,  is_temproal_down = True):
        # conv T is 3x1x1
        return nn.Conv3d(self.out_planes, self.out_planes, 
                        kernel_size=(3, 1, 1), 
                        stride= (self.stride_temporal if is_temproal_down else 1, 1, 1),
                        padding=(self.dilation, 0, 0), 
                        dilation = (self.dilation, 1, 1), bias=False)

    def forward_A(self, x):
        # print('st_a')
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        return x

    def forward_B(self, x):
        # q
        tmp_x = self.conv1(x)
        tmp_x = self.norm1(tmp_x)
        tmp_x = self.relu(tmp_x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        # print('st_b')
        # print(self.conv1, self.conv2)
        # print(tmp_x.shape, x.shape)
        return x + tmp_x

    def forward_C(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        tmp_x = self.conv2(x)
        tmp_x = self.norm2(tmp_x)
        tmp_x = self.relu(tmp_x)
        # print('st_c')
        # print(self.conv1, self.conv2)
        # print(x.shape, tmp_x.shape)
        return x + tmp_x

    def forward(self, x):
        
        if self.combine_type == 'A':
            return self.forward_A(x)
        elif self.combine_type == 'B':
            return self.forward_B(x)
        else:
            return self.forward_C(x)

@BACKBONES.register_module()
class ResNetV1c3dIso(ResNet3dIso):
    """ResNetV1c variant described in [1]_.

    Compared with default ResNet(ResNetV1b), ResNetV1c replaces the 7x7 conv
    in the input stem with three 3x3 convs.

    References:
        .. [1] https://arxiv.org/pdf/1812.01187.pdf
    """

    def __init__(self, **kwargs):
        super(ResNetV1c3dIso, self).__init__(
            deep_stem=True, avg_down=False, **kwargs)


@BACKBONES.register_module()
class ResNetV1d3dIso(ResNet3dIso):
    """ResNetV1d variant described in [1]_.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super(ResNetV1d3dIso, self).__init__(
            deep_stem=True, avg_down=True, **kwargs)
