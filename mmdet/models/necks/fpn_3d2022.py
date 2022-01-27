import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_upsample_layer, build_norm_layer
from mmcv.runner import BaseModule
from typing import Sequence, List
from ..builder import NECKS
from ..utils.ccnet_pure import print_tensor
from ..utils import nan_hook
# from mmdet.models.backbones.convnext_3d import CNextBlock3D


@NECKS.register_module()
class FPN3D2022(BaseModule):
    r"""Feature Pyramid Network.

    modified based on UFPN from https://github.com/MIC-DKFZ/nnDetection/blob/main/nndet/arch/decoder/base.py

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

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
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None, verbose = False, 
                 kernel_size = 5, 
                 upsample_cfg=dict(type='deconv3d', mode=None, use_norm = False, 
                                    kernel_size = (2,2,2), stride = (2,2,2) ),
                 init_cfg=dict(
                     type='Xavier', layer='Conv3d', distribution='uniform'), 
                 is_double_chn = True,     
                ):
        super(FPN3D2022, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.fixed_out_channels = fixed_out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_mode = upsample_cfg.pop('mode', None)
        self.upsample_use_norm = upsample_cfg.pop('use_norm', False)
        self.deconv_cfg = upsample_cfg.copy()
        self.min_out_channels = min_out_channels
        self.kernel_size = kernel_size
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

        self.out_channels = self.compute_output_channels(is_double_chn)
        self.up_ops = self.build_upsample_layers(conv_cfg, norm_cfg = norm_cfg if self.upsample_use_norm else None)
        print(f'[FPN3D] input channels {self.in_channels} out channels {self.out_channels} upmode {self.up_ops[-1]}')
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.verbose = verbose
        for i in range(0, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                self.out_channels[i],
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.out_channels[i],
                self.out_channels[i],
                self.kernel_size,
                padding= (self.kernel_size - 1)//2,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)


        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = fixed_out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    fixed_out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)


    def compute_output_channels(self, is_double_chn = True) -> List[int]:
        """
        Compute number of output channels
        for upper most two levels, the channels can stay the same as the encoder featmap. 
        Returns:
            List[int]: number of output channels for each level
        """
        out_channels = [self.fixed_out_channels] * self.num_ins

        if self.start_level is not None: #2345
            ouput_levels = list(range(self.num_ins)) # encoder outputing levels, 01234
            # filter for levels above decoder levels
            ouput_levels = [ol for ol in ouput_levels if ol < self.start_level]
            # assert max(ouput_levels) < self.start_level, "Can not decrease channels below decoder level"
            for ol in ouput_levels[::-1]: # 1, 0 
                oc = max(self.min_out_channels, self.in_channels[ol]* (2 if is_double_chn else 1 ))
                out_channels[ol] = oc
        return out_channels

    def build_upsample_layers(self, conv_cfg, norm_cfg = None):
        """
        # deconv consumes less gpu memory
        """
        up_ops = nn.ModuleList()
        for i in range(0, self.backbone_end_level):
            if i == 0:
                up_ops.append(nn.Identity())
            else:
                if self.upsample_mode is not None:
                    up = nn.Upsample(scale_factor=2, mode= self.upsample_mode, align_corners=True)
                    if not (self.out_channels[i] == self.out_channels[i - 1]):
                        _conv = ConvModule(self.out_channels[i],
                                            self.out_channels[i - 1], 
                                            1, conv_cfg=conv_cfg, 
                                            norm_cfg = norm_cfg, 
                                            act_cfg=None)
                        up = nn.Sequential(up, _conv)
                else:
                    up = nn.Sequential(build_upsample_layer(
                                        cfg=self.deconv_cfg,
                                        in_channels=self.out_channels[i],
                                        out_channels=self.out_channels[i-1],
                                        bias = False),
                                        nn.Identity() if norm_cfg is None else build_norm_layer(norm_cfg, self.out_channels[i-1])[1],
                                        # nn.ReLU(inplace=True)
                                        )
                up_ops.append(up)
        return up_ops

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # _ = [print_tensor(i, s) for i, s in enumerate(inputs)]
        # build laterals
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            lat_out = lateral_conv(inputs[i])
            # ipdb.set_trace()
            if self.verbose: print_tensor(f'[FPN] Lateral conv level {i}', lat_out )
            laterals.append(lat_out)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1): # 5,4,3,2,1
            # print_tensor(f'[fpn3d ups] {i} {self.up_ops[i]}', laterals[i])

            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            up_feat = self.up_ops[i](laterals[i])
            if self.verbose: print_tensor(f'[FPN] UP level {i}', up_feat )
            laterals[i - 1] = laterals[i - 1] + up_feat

        # build outputs
        outs = []
        # part 1: from original levels
        for i in range(used_backbone_levels):
            out_i = self.fpn_convs[i](laterals[i])
            if self.verbose: print_tensor(f'[FPN] Fuse conv level {i}', out_i )
            outs.append(out_i)

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        # ipdb.set_trace()
        # bb = [print_tensor(f'[FPNeck] inputlevel {i}', o) for i, o in enumerate(inputs)]
        # level_hasnan = [torch.isnan(a).any() for i, a in enumerate(outs)]
        # if self.verbose: 
        #     aa = [print_tensor(f'[FPNeck] level {i}', o) for i, o in enumerate(outs)] #hasnan {level_hasnan[i]}
        # if any(level_hasnan): ipdb.set_trace()
        return tuple(outs)
