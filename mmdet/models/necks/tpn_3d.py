import torch.nn as nn
from mmcv.cnn import ConvModule, build_upsample_layer, build_norm_layer

from typing import Sequence, List
from ..builder import NECKS
from .fpn_3d import FPN3D
from ..backbones.resnet3d import Bottleneck3d
from mmcv.runner import BaseModule
from ..utils.ccnet_pure import print_tensor
import copy

@NECKS.register_module()
class TPN3D(BaseModule):
    r"""Trident Pyramid Network.

    This is an implementation of paper `Trident Pyramid Networks for Object
    Detection <https://arxiv.org/abs/2110.04004>`_.

    modified based on UFPN from https://github.com/MIC-DKFZ/nnDetection/blob/main/nndet/arch/decoder/base.py


    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 fixed_out_channels,
                 num_outs,
                 start_level=2,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 min_out_channels = 8, 
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d'),
                 act_cfg=dict(type='ReLU'), verbose = False, 
                 num_bottleneck = 3, num_tpn = 2, 
                 upsample_cfg=dict(type='deconv3d', mode='trilinear', 
                                    kernel_size = (2,2,2), stride = (2,2,2) ),
                 init_cfg=dict(
                     type='Xavier', layer='Conv3d', distribution='uniform')):
        super(TPN3D, self).__init__(init_cfg)
        assert isinstance(in_channels, list)

        self.in_channels = in_channels
        self.fixed_out_channels = fixed_out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.min_out_channels = min_out_channels
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level

        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        tpn0 = TridentPNconstructor(in_channels,
                                    fixed_out_channels,
                                    num_outs,
                                    start_level,
                                    end_level,
                                    add_extra_convs,
                                    relu_before_extra_convs,
                                    no_norm_on_lateral,
                                    min_out_channels, 
                                    conv_cfg,
                                    norm_cfg, 
                                    act_cfg, verbose, 
                                    upsample_cfg, init_cfg, 
                                    num_bottleneck = num_bottleneck,  
                                    tpn_index=0
                                    )
        tpn_list = [tpn0]
        interm_channels = tpn0.out_channels
        
        for i in range(1, num_tpn):
            tpn_ = TridentPNconstructor(interm_channels,
                                    fixed_out_channels,
                                    num_outs,
                                    start_level,
                                    end_level,
                                    add_extra_convs,
                                    relu_before_extra_convs,
                                    no_norm_on_lateral,
                                    min_out_channels, 
                                    conv_cfg,
                                    norm_cfg, 
                                    act_cfg, verbose, 
                                    upsample_cfg, init_cfg, 
                                    num_bottleneck = num_bottleneck, 
                                    tpn_index= i
                                    )
            tpn_list.append(tpn_)
        
        self.tpn_block = nn.ModuleList(tpn_list)
        
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        x = self.tpn_block[0](inputs)
        for tpn_ in self.tpn_block[1:]:
            x = tpn_(x)
        return x
        

class TridentPNconstructor(BaseModule):

    def __init__(self, 
                 in_channels,
                 fixed_out_channels,
                 num_outs,
                 start_level=2,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 min_out_channels = 8, 
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None, verbose = False, 
                 upsample_cfg=dict(type='deconv3d', mode='trilinear', 
                                    kernel_size = (2,2,2), stride = (2,2,2) ),
                init_cfg=dict(type='Xavier', layer='Conv3d', distribution='uniform'), 
                num_bottleneck = 2, 
                tpn_index = 0
                     ):
        super(TridentPNconstructor, self).__init__(init_cfg)
        
        self.top_down_fpn = FPN3D(in_channels, 
                                fixed_out_channels, 
                                num_outs,
                                start_level,
                                end_level,
                                add_extra_convs,
                                relu_before_extra_convs,
                                no_norm_on_lateral,
                                min_out_channels, 
                                conv_cfg,
                                norm_cfg,
                                act_cfg = None, verbose = verbose, 
                                upsample_cfg = copy.deepcopy(upsample_cfg),
                                init_cfg = init_cfg, 
                                is_double_chn = tpn_index == 0
                                )

        self_in_channels = self.top_down_fpn.out_channels
        self.self_procession = TPNSelfProcess(self_in_channels, 
                                             num_bottleneck, 
                                             start_level, 
                                             init_cfg, 
                                             conv_cfg = conv_cfg, 
                                             norm_cfg = norm_cfg,
                                             act_cfg = act_cfg, 
                                             inflate_style = '3x3x3')

        self.bottom_up_fpn = TPNBottomUP(self_in_channels, 
                                        start_level = start_level, 
                                        conv_cfg = conv_cfg, 
                                        norm_cfg = norm_cfg,
                                        act_cfg = act_cfg,
                                        init_cfg = init_cfg)
        
        self.self_process_final = TPNSelfProcess(self_in_channels, 
                                             1, 
                                             start_level, 
                                             init_cfg, 
                                             conv_cfg = conv_cfg, 
                                             norm_cfg = norm_cfg,
                                             act_cfg = act_cfg, 
                                             inflate_style = '3x3x3')
        self.out_channels = self_in_channels

    def forward(self, inputs):

        print_level1by1 = lambda n, inputs: [print_tensor(f'{n} l{i}', t) for i, t in enumerate(inputs)]

        # print_level1by1('input', inputs)
        top_down_outs = self.top_down_fpn(inputs)
        # print_level1by1('topdown', top_down_outs)
        self_outs = self.self_procession(top_down_outs)
        # print_level1by1('self', self_outs)
        bottom_up_outs = self.bottom_up_fpn(self_outs)
        # print_level1by1('bottomup', bottom_up_outs)
        final_outs = self.self_process_final(bottom_up_outs)
        # print_level1by1('final', final_outs)
        return final_outs

class TPNSelfProcess(BaseModule):
    
    def __init__(self, in_channels, num_bottleneck, start_level = 0, 
                 init_cfg=None, **conv_kwargs):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.num_bottleneck = num_bottleneck
        self.start_level = start_level

        self.self_process_list = nn.ModuleList()
        for i, in_chn in enumerate(in_channels):

            if i < start_level:
                ops_i = nn.Identity()
            else:
                
                ops_i = nn.Sequential(*[Bottleneck3d(in_chn, in_chn // 4, **conv_kwargs)
                                      for i in range(self.num_bottleneck)])
            
            self.self_process_list.append(ops_i)
        
    def forward(self, inputs):
        outs = []
        for i, ips in enumerate(inputs):
            out_ = self.self_process_list[i](ips)
            outs.append(out_)
        return outs
    
class TPNBottomUP(BaseModule):

    """
    
    from P3 ~ P7
    high resolution (low channel) to low resolution (high channel)
    

    """

    def __init__(self, 
                in_channels,
                start_level = 2, 
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None, verbose = False, 
                init_cfg=dict(type='Xavier', layer='Conv3d', distribution='uniform'), 
                ):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.start_level = start_level

        self.fuse_op_list = []
        self.num_levels = len(in_channels)
        # in_channels_reversed = in_channels[::-1]
        for i, in_chn in enumerate(in_channels): # 0 1 2 3 4 5
            if i < start_level:
                ops_i = nn.Identity()
                prev_chn = None
            else:
                prev_chn = in_channels[i - 1]
                ops_i = nn.Sequential(
                            ConvModule(prev_chn, prev_chn//4, 1, 
                                        conv_cfg = conv_cfg, norm_cfg = norm_cfg, act_cfg = act_cfg), 
                            ConvModule(prev_chn//4, prev_chn//4, 3, stride = 2, padding= 1, 
                                        conv_cfg = conv_cfg, norm_cfg = norm_cfg, act_cfg = act_cfg), 
                            ConvModule(prev_chn//4, in_chn, 1,  
                                        conv_cfg = conv_cfg, norm_cfg = norm_cfg, act_cfg = act_cfg), 
                            )
            # print(f'[TPN] BU lix{i} inchn{in_chn} prev{prev_chn}') # i = 012345; in_chn = 128 128 128 64 32
            self.fuse_op_list.append(ops_i)
        
        self.fuse_op_list = nn.ModuleList(self.fuse_op_list)
        
    def forward(self, inputs):

        outs  = []
        feat_curruncy = inputs[0]
        for i, ips in enumerate(inputs): # 0 1 2 3 4
            # print_tensor(f'\n[TPN] forward lix{i} prev', feat_curruncy)
            # print_tensor(f'[TPN] forward lix{i} ips', ips)
            # print(self.fuse_op_list[i])          
            if i < self.start_level:
                feat_curruncy = self.fuse_op_list[i](ips)
            else:
                feat_curruncy = ips + self.fuse_op_list[i](feat_curruncy)
            outs.append(feat_curruncy)

        return outs


def compute_output_channels(in_channels,  start_level, 
                            fixed_out_channels, min_out_channels,) -> List[int]:
    """
    Compute number of output channels
    for upper most two levels, the channels can stay the same as the encoder featmap. 
    Returns:
        List[int]: number of output channels for each level
    """
    num_ins = len(in_channels)
    out_channels = [fixed_out_channels] * num_ins

    if start_level is not None: #2345
        ouput_levels = list(range(num_ins)) # encoder outputing levels
        # filter for levels above decoder levels
        ouput_levels = [ol for ol in ouput_levels if ol < start_level]
        assert max(ouput_levels) < start_level, "Can not decrease channels below decoder level"
        for ol in ouput_levels[::-1]: # 1, 0 
            oc = max(min_out_channels, in_channels[ol]*2)
            out_channels[ol] = oc
    return out_channels

def build_upsample_layers(backbone_end_level, out_channels, 
                            conv_cfg, deconv_cfg, 
                            upsample_mode):
    up_ops = nn.ModuleList()
    for i in range(0, backbone_end_level):
        if i == 0:
            up_ops.append(None)
        else:
            if upsample_mode is not None:
                up = nn.Upsample(scale_factor=2, mode= upsample_mode)
                if not (out_channels[i] == out_channels[i - 1]):
                    _conv = ConvModule(out_channels[i], 
                                        out_channels[i - 1], 
                                        1, 
                                        conv_cfg= conv_cfg, 
                                        norm_cfg = None, act_cfg=None)
                    up = nn.Sequential(up, _conv)
            else:
                up = nn.Sequential(build_upsample_layer(
                                            cfg=deconv_cfg,
                                            in_channels=out_channels[i],
                                            out_channels=out_channels[i-1],
                                            bias = False),
                                            # build_norm_layer(norm_cfg, up_channels)[1],
                                            # nn.ReLU(inplace=True)
                                        )
            up_ops.append(up)
    return up_ops