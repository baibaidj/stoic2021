# Copyright (c) OpenMMLab. All rights reserved.
from .brick_wrappers import AdaptiveAvgPool2d, adaptive_avg_pool2d
from .builder import build_linear_layer, build_transformer
from .ckpt_convert import pvt_convert
from .conv_upsample import ConvUpsample
from .csp_layer import CSPLayer
from .gaussian_target import gaussian_radius, gen_gaussian_target
from .inverted_residual import InvertedResidual
from .make_divisible import make_divisible
from .misc import interpolate_as, nan_hook
from .normed_predictor import NormedConv2d, NormedLinear
from .positional_encoding import (LearnedPositionalEncoding,
                                  SinePositionalEncoding)
from .res_layer import ResLayer, SimplifiedBasicBlock, ResLayer3D, ResLayerIso
from .se_layer import SELayer
from .transformer import (DetrTransformerDecoder, DetrTransformerDecoderLayer,
                          DynamicConv, PatchEmbed, Transformer, nchw_to_nlc,
                          nlc_to_nchw)

from .transformer3d import (PatchEmbed3D, nchwd_to_nlc, nlc_to_nchwd)
from .modulated_deform_conv import DeformConv3d, ModulatedDeformConv3d
print_tensor = lambda n, x: print(n, type(x), x.dtype, x.shape, x.min(), x.max())
chn2last_order = lambda x: tuple([0, *[a + 2 for a in range(x)],  1])


from mmcv.cnn import CONV_LAYERS
CONV_LAYERS.register_module('DCN3dv1', module=DeformConv3d)
CONV_LAYERS.register_module('DCN3dv2', module=ModulatedDeformConv3d)

__all__ = [
    'ResLayer', 'gaussian_radius', 'gen_gaussian_target',
    'DetrTransformerDecoderLayer', 'DetrTransformerDecoder', 'Transformer',
    'build_transformer', 'build_linear_layer', 'SinePositionalEncoding',
    'LearnedPositionalEncoding', 'DynamicConv', 'SimplifiedBasicBlock',
    'NormedLinear', 'NormedConv2d', 'make_divisible', 'InvertedResidual',
    'SELayer', 'interpolate_as', 'ConvUpsample', 'CSPLayer',
    'adaptive_avg_pool2d', 'AdaptiveAvgPool2d', 'PatchEmbed', 'nchw_to_nlc',
    'nlc_to_nchw', 'pvt_convert', 

    'ResLayer3D', 'ResLayerIso', 'PatchEmbed3D', 'nlc_to_nchwd', 'nchwd_to_nlc', 'nan_hook', 
    'DeformConv3d', 'ModulatedDeformConv3d'
]
