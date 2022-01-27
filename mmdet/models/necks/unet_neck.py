import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..builder import NECKS
from ..utils.unet_bricks import CenterBlock, DecoderBlock_Vnet

@NECKS.register_module()
class UnetNeck3D(nn.Module):
    """Unet.

    This Neck is the implementation of `Unet`

    Args:

    """

    def __init__(self, 
            in_channels=(256, 512, 1024, 2048),
            channels=(256, 128, 64, 32),
            drop_out_ratio=0.1,
            conv_cfg=None,
            norm_cfg=None,
            skip_levels = 3, 
            align_corners = False,
            sece_blocks = None, 
            aug_top_feat = False,
            is_bottleneck = False, 
            is_p3d = False,
            nlblocks = (1,1,1),
            nl_cfg = dict(type = 'naive',
                         fusion_type = 'add', 
                         key_source = 'self', 
                         psp_size = None,
                         reduction = 8, out_projection = True)
            ):
        super(UnetNeck3D, self).__init__()

        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.align_corners = align_corners


        assert isinstance(self.in_channels, (list, tuple)) 
        assert isinstance(self.channels, (list, tuple))
        
        # last layer for seg

        self.unetdecoder = UnetDecoder3D(
            encoder_channels = self.in_channels,  # (256, 512, 1024, 2048)
            decoder_channels = self.channels,  #  (256, 128, 64, 32),
            conv_cfg = self.conv_cfg,
            norm_cfg = self.norm_cfg,
            skip_levels = skip_levels, 
            sece_blocks=sece_blocks,
            align_corners = self.align_corners,
            aug_top_feat = aug_top_feat,
            is_bottleneck = is_bottleneck, 
            is_p3d = is_p3d,
            nlblocks = nlblocks,
            nl_cfg = nl_cfg
            # nl_fusion_type = nl_fusion_type,
            # nl_psp_size = nl_psp_size,
            # nl_key_source = nl_key_source,
            # cc_post_conv2 = cc_post_conv2,
            # cc_reduction = cc_reduction
        )
        self.init_weights()

    def init_weights(self):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)


    def forward(self, inputs):
        """Forward function."""

        # for i, x in enumerate(inputs):
        #     print_tensor('Resnet3d %d stage' %i, x)

        feat_maps = self.unetdecoder(*inputs)

        return feat_maps # high2low resolution
    
    def forward_train1(self, *features):
        # print('nb features', len(features))
        # features = features[::-1]  # reverse channels to start from head of encoder
        # [print(a.shape) for a in features]
        # print('nb features', len(features))
        features = features[::-1]  # reverse channels to start from head of encoder
        # [print(a.shape) for a in features]
        head = features[0] # last to first, 2048
        skips = features[1:] # 1024, 512, 256, #64, 3 features[len(features):0:-1]

        x = self.unetdecoder.center(head)
        # pdb.set_trace()
        # outputs = [x]
        for i in range(2):
            decoder_block = self.unetdecoder.blocks[i]
            skip = skips[i] if self.unetdecoder.skip_channels[i] else None #len(skips)==4
            # print_tensor('from down %d' %i, x)
            # print_tensor('skip %d' %i, skip)
            x = decoder_block(x, skip)
            # outputs.append(x)
        return x

    def forward_train2(self, x, *features):
        # print('nb features', len(features))
        features = features[::-1]  # reverse channels to start from head of encoder
        # [print(a.shape) for a in features]
        # head = features[0] # last to first, 2048
        skips = features[1:] # 1024, 512, 256, #64, 3
        # x = self.center(head)
        outputs = [x]
        for i in range(2, self.unetdecoder.nb_decoder):
            decoder_block = self.unetdecoder.blocks[i]
            skip = skips[i] if self.unetdecoder.skip_channels[i] else None #len(skips)==4
            # print_tensor('from down %d' %i, x)
            # print_tensor('skip %d' %i, skip)
            x = decoder_block(x, skip)
            outputs.append(x)

        return outputs[::-1]
        # pass

class UnetDecoder3D(nn.Module):
    def __init__(
            self,
            encoder_channels, # (256, 512, 1024, 2048) at most 4 output layers
            decoder_channels,  #  (256, 128, 64, 32),
            conv_cfg = None,
            norm_cfg = None, 
            sece_blocks=(None, None, None, None),
            skip_levels = 3, 
            align_corners = False,
            aug_top_feat = False,
            is_p3d = False,
            is_bottleneck = False,
            nlblocks = (True, True, True, True),  
            nl_cfg = dict(type = 'naive',
                         fusion_type = 'add', 
                         key_source = 'self', 
                         psp_size = None,
                         reduction = 8, out_projection = True)
            # nl_fusion_type = 'add',
            # nl_psp_size = None,
            # nl_key_source = 'self',
            # cc_post_conv2 = False,
            # cc_reduction = 8,


    ):
        super(UnetDecoder3D, self).__init__()
        assert nl_cfg['type'] in ('naive', 'cca')
        decoder_ops = DecoderBlock_Vnet #if use_shuffle else DecoderBlock_NL 
        # self.is_bottleneck = is_bottleneck
        self.sece_blocks = sece_blocks
        self.nl_cfg = nl_cfg
        self.nb_decoder = len(decoder_channels)
        self.skip_levels = skip_levels
        # decoder_channels = decoder_channels[:skip_levels]
        nlblocks = list(nlblocks) + [0] * self.nb_decoder
        sece_blocks = list(sece_blocks) + [None] * self.nb_decoder


        in_channels, skip_channels, out_channels = self.prepare_channels(encoder_channels, decoder_channels, 
                                                    skip_levels = skip_levels,
                                                    aug_top_feat = aug_top_feat)

        self.skip_channels = skip_channels
        
        print('in:\t', in_channels)
        print('skip:\t', skip_channels)
        print('out:\t', out_channels)


        if aug_top_feat:
            self.center = CenterBlock(
                self.head_channels, conv_cfg, norm_cfg
            )
        else:
            self.center = nn.Identity()
        # combine decoder keyword arguments
        kwargs = dict(conv_cfg = conv_cfg, norm_cfg= norm_cfg,
                    align_corners = align_corners, is_p3d = is_p3d, is_bottleneck = is_bottleneck,
                    nl_cfg = nl_cfg, #verbose = True,
                    )
        # print(kwargs)
        blocks = [
            decoder_ops(in_ch, skip_ch, out_ch, is_nl, level, use_sece, **kwargs)
            for in_ch, skip_ch, out_ch, is_nl, level, use_sece in 
            zip(in_channels, skip_channels, out_channels, nlblocks, list(range(self.skip_levels)), sece_blocks)
        ]
        self.blocks = nn.ModuleList(blocks)

    def prepare_channels(self, encoder_channels, decoder_channels, skip_levels = 3, aug_top_feat = False):
        # nn.Identity(), 3
        # nn.Sequential(self.conv1, self.bn1, self.relu), 1 start here, 1/2, 64
        # nn.Sequential(self.maxpool, self.layer1), 1/4
        # self.layer2, 1/8
        # self.layer3, 1/16
        # self.layer4,
        # (3, 64, 256, 512, 1024, 2048),
        # (1, 1/2, 1/4, 1/8, 1/16, 1/32)
        # remove first skip with same spatial resolution
        # encoder_channels = [encoder_channels[-1-a] for a in range(min(5, len(encoder_channels)))] 

        self.head_channels = encoder_channels[-1] # 2048 

        print('encoder channels', encoder_channels)
        # reverse channels to start from head of encoder
        encoder_channels = list(encoder_channels)[:-skip_levels-2:-1] + [0]*self.nb_decoder 
        print('encoder channels', encoder_channels)

        # this is fixed
        skip_channels = [encoder_channels[1+i] for i in range(self.nb_decoder)] # 1024, 512, 256, 64, 0

        if aug_top_feat:
            in_channels, out_channels = [self.head_channels] + [0] * (self.nb_decoder -1), [0] * self.nb_decoder
            for i in range(self.nb_decoder):
                out_channels[i] = in_channels[i]//2 + skip_channels[i]
                if i + 1 <= self.nb_decoder -1: 
                    in_channels[i+1] = out_channels[i]
        else:
            in_channels = [self.head_channels] + list(decoder_channels[:-1]) # 2048 256 128 64 32
            out_channels = decoder_channels # 256 128 64 32 16

        return in_channels, skip_channels, out_channels

    def forward(self, *features):
        
        # print('nb features', len(features))
        features = features[::-1]  # reverse channels to start from head of encoder
        # [print(a.shape) for a in features]
        head = features[0] # last to first, 2048
        skips = features[1:] # 1024, 512, 256, #64, 3
        x = self.center(head)
        outputs = [x]
        for i in range(self.nb_decoder):
            decoder_block = self.blocks[i]
            skip = skips[i] if self.skip_channels[i] else None #len(skips)==4
            # print_tensor('from down %d' %i, x)
            # print_tensor('skip %d' %i, skip)
            x = decoder_block(x, skip)
            outputs.append(x)
        
        return outputs[::-1]
print_tensor = lambda n, x: print(n, type(x), x.shape, x.min(), x.max())