import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from ..builder import HEADS
from .decode_head_med import BaseDecodeHeadMed, print_tensor
from ..utils.implicit_semantic_data_aug import ISDALoss
from ...utils.resize import bnchw2bchw
# from mmcv.runner import auto_fp16

@HEADS.register_module()
class FCNHead3D(BaseDecodeHeadMed):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        is_use_isda (boo): if use implicit semantic data augmentation
        isda_lambda (float) : 'The hyper-parameter \lambda_0 for ISDA, select from {1, 2.5, 5, 7.5, 10}. '
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 is_use_isda = False, 
                 isda_lambda = 2.5,
                 start_iters = 1,
                 max_iters = 4e5,
                 is_isda_3d = False,
                 verbose = False,
                 acc_gt_index = 0,
                 **kwargs):
        assert isinstance(num_convs, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.acc_gt_index = acc_gt_index
        super(FCNHead3D, self).__init__(**kwargs)
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for _ in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
                
        self.is_isda_3d = is_isda_3d
        self.verbose = verbose
        self.is_use_isda = is_use_isda
        self.isda_lambda = isda_lambda
        self._iter = start_iters
        self._max_iters = max_iters
        if is_use_isda:
            self.isda_augmentor = ISDALoss(self.final_channel, self.num_classes, is_3d = self.is_isda_3d)

    def forward(self, inputs):
        """Forward function."""
        # ratio = args.lambda_0 * global_iteration / args.num_steps # training progress as percentage 
        if self.verbose: 
            for i, ip in enumerate(inputs): print_tensor(f'[FCNHead] input {i}', ip)
        x = self._transform_inputs(inputs)
        feat_map = self.convs(x) if self.num_convs > 0 else x
        if self.concat_input:
            feat_map = self.conv_cat(torch.cat([x, feat_map], dim=1))
            
        output = self.cls_seg(feat_map)
        if self.verbose: 
            print_tensor('[FCNHead] finalfeat', feat_map)
            print_tensor('[FCNHead] fcnout', output)
        if self.is_use_isda and self.training :
            return output, feat_map.detach() 
        else:
            return output

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

        # print_tensor('2channelgt', gt_semantic_seg)
        if self.gt_index is not None:
            gt = gt_semantic_seg[:, self.gt_index : self.gt_index + 1, ...]
        else: gt = gt_semantic_seg
        # print_tensor('1channelgt', gt)
        gt, *_ = bnchw2bchw(gt, train_cfg.get('use_tsm', False))
        if self.is_use_isda:
            x = self._transform_inputs(inputs)
            feat_map = self.convs(x) if self.num_convs > 0 else x
            if self.concat_input:
                feat_map = self.conv_cat(torch.cat([x, feat_map], dim=1))
            if self.dropout is not None:
                feat_map = self.dropout(feat_map)
            ratio = self.isda_lambda * self._iter / self._max_iters # training progress as percentage 
            conv_seg_list =  self.conv_seg if self.is_multi_conv_seg else [self.conv_seg]
            seg_logit_list = []
            for s, conv_seg in enumerate(conv_seg_list):
                seg_raw = conv_seg(feat_map)
                # pdb.set_trace()
                gt_s = gt[:, :, s] if self.is_multi_conv_seg else gt
                seg_aug = self.isda_augmentor(feat_map.detach(), conv_seg, seg_raw, gt_s, ratio) #
                seg_logit_list.append(seg_aug)
            seg_logits = torch.cat(seg_logit_list, dim = 1)
            self._iter += 1
        else:
            seg_logits = self.forward(inputs)
        if self.verbose: 
            print_tensor('[FCNTrain] logits', seg_logits)
            print_tensor('[FCNTrain] Gtruth', gt)
        losses = self.losses(seg_logits, gt, self.acc_gt_index)

        return losses

    def simple_test(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 5D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 7),
                where 57represent (tl_x, tl_y, tl_z, br_x, br_y, br_z, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        outs = self.forward(feats)
        return outs