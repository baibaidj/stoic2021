import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import (ConvModule, build_activation_layer,
                      build_conv_layer, build_norm_layer, build_plugin_layer,
                      constant_init, kaiming_init) 
from mmcv.utils import _BatchNorm
from torch.nn.modules.utils import _ntuple, _triple
from mmcv.runner import BaseModule, load_checkpoint

from ...utils import get_root_logger
from ..builder import BACKBONES
import torch, copy, pdb

print_tensor = lambda n, x: print(n, type(x), x.dtype, x.shape, x.min(), x.max())

class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv3d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv3d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        # BCHW
        x = F.avg_pool3d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x) # BCHW
        x = x.view(-1, self.input_channels, 1, 1, 1) # BHW, C,1,1
        return inputs * x

class RepVGGBlock(BaseModule):

    def __init__(self, inplanes, planes, kernel_size=3, 
                 stride=1, padding=0, dilation=1, groups=1, 
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d'),
                 act_cfg=dict(type='ReLU'),
                 deploy=False, use_se=False, 
                 padding_mode='zeros',
                 init_cfg = None
                 ):
        super(RepVGGBlock, self).__init__(init_cfg=init_cfg)
        self.deploy = deploy
        self.groups = groups
        self.inplanes = inplanes
        self.planes = planes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.is_3d = '3' in self.conv_cfg['type']

        assert padding == 1

        # padding_11 = padding - kernel_size // 2
        kernel2pad = lambda a : (a - 1)//2

        self.nonlinearity = nn.ReLU()

        if use_se:
            self.se = SEBlock(planes, internal_neurons=planes // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = build_conv_layer(conv_cfg, inplanes, planes, kernel_size, 
                                                stride = stride, padding = kernel2pad(kernel_size),
                                                dilation=dilation, groups=groups, bias = True)

        else:
            self.rbr_identity = build_norm_layer(self.norm_cfg, inplanes)[1] \
                                         if planes == inplanes and stride == 1 else None

            self.rbr_dense = ConvModule(inplanes, planes, kernel_size, stride, padding,
                                        dilation=dilation, groups=groups, bias = False, 
                                        conv_cfg = self.conv_cfg, norm_cfg= self.norm_cfg)
            self.rbr_1x1 = ConvModule(inplanes, planes, 1, stride, 0, 
                                        dilation=dilation, groups=groups, bias = False, 
                                        conv_cfg = self.conv_cfg, norm_cfg= self.norm_cfg)
            # print('RepVGG Block, identity = ', self.rbr_identity)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))


    #   Optional. This improves the accuracy and facilitates quantization.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_custom_L2()
    #       optimizer.zero_grad()
    #       loss.backward()
    def get_custom_L2(self):
        if self.is_3d:
            K3 = self.rbr_dense.conv.weight
            K1 = self.rbr_1x1.conv.weight
            t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1, 1).detach()
            t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1, 1).detach()

            l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
            eq_kernel = K3[:, :, 1:2, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
            l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        else:

            K3 = self.rbr_dense.conv.weight
            K1 = self.rbr_1x1.conv.weight
            t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
            t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

            l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
            eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
            l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle



#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            # pad left and right (first 2), times number of spatial dim (second 3 or 2)
            pad = [1] * 2 * (3 if self.is_3d else 2)
            return torch.nn.functional.pad(kernel1x1, pad)

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, (nn.Sequential, ConvModule)):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, (nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))
            if not hasattr(self, 'id_tensor'):
                input_dim = self.inplanes // self.groups
                kernel_size = [self.inplanes, input_dim] + [3] * (3 if self.is_3d else 2)
                kernel_value = torch.zeros(kernel_size, dtype=torch.float)
                for i in range(self.inplanes):
                    if self.is_3d: 
                        kernel_value[i, i % input_dim, 1, 1, 1] = 1
                    else:
                        kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = kernel_value.to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t_shape = [-1, 1, 1, 1, 1] if self.is_3d else [-1, 1, 1, 1]
        t = (gamma / std).reshape(*t_shape)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        conv_rep = nn.Conv3d if self.is_3d else nn.Conv2d
        self.rbr_reparam = conv_rep(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)

        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')

@BACKBONES.register_module()
class RepVGG(BaseModule):

    optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    g2_map = {l: 2 for l in optional_groupwise_layers}
    g4_map = {l: 4 for l in optional_groupwise_layers}

    arch_settings = {
        'a0': dict(stage_blocks = (2, 4, 14, 1), width_multiplier=[0.75, 0.75, 0.75, 2.5], ),
        'a1': dict(stage_blocks = (2, 4, 14, 1), width_multiplier=[1, 1, 1, 2.5]),
        'a2': dict(stage_blocks = (2, 4, 14, 1), width_multiplier=[1.5, 1.5, 1.5, 2.75]),
        'b0': dict(stage_blocks = [4, 6, 16, 1], width_multiplier=[1, 1, 1, 2.5]),
        'b1': dict(stage_blocks = [4, 6, 16, 1], width_multiplier=[2, 2, 2, 4]),
        'b2': dict(stage_blocks = [4, 6, 16, 1], width_multiplier=[2.5, 2.5, 2.5, 5]),
        'b3': dict(stage_blocks = [4, 6, 16, 1], width_multiplier=[3, 3, 3, 5]),

        'b0s': dict(stage_blocks = [4, 6, 9, 1], width_multiplier=[1, 1, 1, 2.5]),
        'b1s': dict(stage_blocks = [4, 6, 9, 1], width_multiplier=[2, 2, 2, 4]),
        'b2s': dict(stage_blocks = [4, 6, 9, 1], width_multiplier=[2.5, 2.5, 2.5, 5]),
        'b3s': dict(stage_blocks = [4, 6, 9, 1], width_multiplier=[3, 3, 3, 5]),

        'b0g2': dict(stage_blocks = [4, 6, 9, 1], width_multiplier=[1, 1, 1, 2.5], override_groups_map = g2_map),
        'b1g2': dict(stage_blocks = [4, 6, 16, 1], width_multiplier=[2, 2, 2, 4], override_groups_map = g2_map),
        'b1g4': dict(stage_blocks = [4, 6, 16, 1], width_multiplier=[2, 2, 2, 4], override_groups_map = g4_map),
        'b2g2': dict(stage_blocks = [4, 6, 16, 1], width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map = g2_map ),
        'b2g4': dict(stage_blocks = [4, 6, 16, 1], width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map = g4_map ),
        'b3g2': dict(stage_blocks = [4, 6, 16, 1], width_multiplier=[3, 3, 3, 5], override_groups_map = g2_map ),
        'b3g4': dict(stage_blocks = [4, 6, 16, 1], width_multiplier=[3, 3, 3, 5], override_groups_map = g4_map ),

        'b0sd': dict(stage_blocks = [3, 4, 6, 3], width_multiplier=[1, 1, 1, 1]), # d stands for dense prediction
        'b0sg2d': dict(stage_blocks = [3, 4, 6, 3], width_multiplier=[1, 1, 1, 1], override_groups_map = g2_map),
        'b0sg4d': dict(stage_blocks = [3, 4, 6, 3], width_multiplier=[1, 1, 1, 1], override_groups_map = g4_map),


    }
    def __init__(self, depth, in_channels=3, 
                    base_channels=16, 
                    num_stages=4, 
                    strides=(1, 2, 2, 2),  
                    dilations=(1, 1, 1, 1), 
                    out_indices=(0, 1, 2, 3), 
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=dict(type='BN3d', requires_grad=True),
                    act_cfg=None,
                    deploy=False, use_se=False, 
                    verbose = False, 
                    init_cfg = None):
        super(RepVGG, self).__init__(init_cfg = init_cfg)

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for RepVGG')

        self.depth = depth
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.out_indices = out_indices
        # assert max(out_indices) < num_stages
        self.strides = strides
        self.dilations = dilations
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.verbose = verbose
        self.fp16_enabled = False

        stage_blocks = self.arch_settings[depth]['stage_blocks']
        width_multiplier  = self.arch_settings[depth]['width_multiplier']
        override_groups_map = self.arch_settings[depth].get('override_groups_map', None)

        self.stage_blocks = stage_blocks[:num_stages]
        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_se = use_se

        assert 0 not in self.override_groups_map

        self.in_planes = min(base_channels, int(base_channels * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels, self.in_planes//2, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, 
                                  kernel_size=3, stride=1, padding=1,  deploy=self.deploy, use_se=self.use_se)
        self.stage0_1 = RepVGGBlock(self.in_planes//2, self.in_planes, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, 
                                  kernel_size=3, stride=1, padding=1,  deploy=self.deploy, use_se=self.use_se)        

        if self.verbose: print('[RepVGG] stage0', in_channels, self.in_planes)
        self.res_layers = []  
        self.cur_layer_idx = 1
        for i, num_blocks in enumerate(self.stage_blocks):
            out_planes = int(base_channels*(2**(1+i)) * width_multiplier[i])
            if self.verbose: print(f'[RepVGG] stage {i} blocks {num_blocks}',
                                     self.in_planes, out_planes)
            res_layer = self._make_stage(out_planes, num_blocks, stride=2)
            layer_name = f'stage{i+1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

    def _make_stage(self, planes, stage_blocks, stride):
        strides = [stride] + [1]*(stage_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(self.in_planes, planes, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, 
                                     kernel_size=3,  stride=stride, padding=1, groups=cur_groups, 
                                     deploy=self.deploy, use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        outs = [x]
        x = self.stage0(x)
        x = self.stage0_1(x)
        outs.append(x)

        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            # print(i, res_layer)
            x = res_layer(x)
            if self.verbose: print_tensor(f'l{i+1} {layer_name}', x)
            outs.append(x)
        # print('out indices', self.out_indices)
        return tuple([outs[i] for i in self.out_indices])

    def init_weights_old(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')



#   Use this for converting a RepVGG model or a bigger model with RepVGG as its component
#   Use like this
#   model = create_RepVGG_A0(deploy=False)
#   train model or load weights
#   repvgg_model_convert(model, save_path='repvgg_deploy.pth')
#   If you want to preserve the original model, call with do_copy=True

#   ====================== for using RepVGG as the backbone of a bigger model, e.g., PSPNet, the pseudo code will be like
#   train_backbone = create_RepVGG_B2(deploy=False)
#   train_backbone.load_state_dict(torch.load('RepVGG-B2-train.pth'))
#   train_pspnet = build_pspnet(backbone=train_backbone)
#   segmentation_train(train_pspnet)
#   deploy_pspnet = repvgg_model_convert(train_pspnet)
#   segmentation_test(deploy_pspnet)
#   =====================   example_pspnet.py shows an example

def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model
