
from ...utils.resize import resize_3d, bnchw2bchw
from .decode_head import *
import pdb

print_tensor = lambda n, x: print(n, type(x), x.dtype, x.shape, x.min(), x.max())

chn2last_order = lambda x: tuple([0, *[a + 2 for a in range(x)],  1])

class BaseDecodeHeadMed(BaseDecodeHead):
    """Base class for BaseDecodeHead 3d.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int): The label index to be ignored. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self, in_channels, channels, *args, 
                        num_slice2pred = 1, gt_index = None, 
                        **kwargs):
        self.num_slice2pred = num_slice2pred
        self.gt_index = gt_index
        final_channel = channels if type(channels) is int else channels[-1]
        args = (in_channels, final_channel) + args
        actual_num_classes = kwargs['num_classes']
        if kwargs['num_classes'] is None: kwargs['num_classes'] = 1
        super(BaseDecodeHeadMed, self).__init__(*args, **kwargs)

        self.cls_out_channels = actual_num_classes # max(actual_num_classes, 2)

        self.num_classes = actual_num_classes
        self.is_3d = False if self.conv_cfg is None else (True if '3d' in self.conv_cfg.get('type', '').lower() else False)
        self.get_shape = lambda x: x.shape[2:]
        self.get_mode = lambda : 'trilinear' if self.is_3d else 'bilinear'

        self.channels = channels
        self.final_channel = final_channel
        self.conv_final = nn.Conv3d if self.is_3d else nn.Conv2d

        if self.num_classes is None: 
            self.conv_seg = nn.Identity()
        else:
            self.conv_seg = self.conv_final(self.final_channel, self.cls_out_channels, kernel_size=1)

        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout3d(self.dropout_ratio) if self.is_3d else nn.Dropout2d(self.dropout_ratio)
        else:
            self.dropout = None
    @property
    def is_multi_conv_seg(self):
        return self.num_slice2pred > 1

    def init_weights(self):
        """Initialize weights of classification layer."""

        if isinstance(self.conv_seg, torch.nn.ModuleList):
            for conv_seg in self.conv_seg:
                normal_init(conv_seg, mean=0, std=0.1)
        else:
            if isinstance(self.conv_seg, (nn.Conv2d, nn.Conv3d)):
                normal_init(self.conv_seg, mean=0, std=0.1)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        if isinstance(self.conv_seg, torch.nn.ModuleList):
            out_list = []
            for conv_seg in self.conv_seg:
                out_i = conv_seg(feat)
                out_list.append(out_i)
            output = torch.cat(out_list, axis = 1)
        else:
            output = self.conv_seg(feat)
        return output


    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize_3d(
                    input=x,
                    size=self.get_shape(inputs[0]),
                    mode=self.get_mode(),
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            # print('decoder input %d, need indexes %s' %(len(inputs), self.in_index))
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs
    
    def split_channels2slice(self, seg_logit):
        if self.num_slice2pred > 1:
            chunk_list = torch.split(seg_logit, self.num_slice2pred, dim = 1)
            seg_logit = torch.stack(chunk_list, dim = 1) # B, C*S, H, W >> B, C, S, H, W
        return seg_logit

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, acc_gt_index = 0):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize_3d(
            input=seg_logit,
            size=self.get_shape(seg_label),
            mode=self.get_mode(),
            align_corners=self.align_corners)
        
        seg_logit = self.split_channels2slice(seg_logit)
        # print_tensor('LOSS: seg logit', seg_logit)
        # print_tensor('LOSS: seg label', seg_label)
        assert seg_label.dim() == seg_logit.dim(), f'seg_label should be 5 dim but the shape is {seg_label.shape}'
        # seg_label should be B, 1, S, H, W

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        
        if seg_label.shape[1] == 1: seg_label = seg_label.squeeze(1)

        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        if self.is_3d and seg_label.dim() == 5: seg_label = seg_label[:, acc_gt_index, ...]
        dim_order = chn2last_order(seg_logit.dim() - 2)
        loss['acc_seg'] = accuracy(seg_logit.permute(*dim_order).reshape(-1, self.cls_out_channels), 
                                    seg_label.reshape(-1))
        return loss

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.gt_index is not None: gt_semantic_seg = gt_semantic_seg[:, self.gt_index : self.gt_index + 1, ...]
        gt_semantic_seg, *_ = bnchw2bchw(gt_semantic_seg, train_cfg.get('use_tsm', False))
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses