import torch
import torch.nn as nn
from mmcv.cnn import normal_init, kaiming_init, constant_init, trunc_normal_init
from mmcv.runner import BaseModule

from ..builder import HEADS
from .cls_head import ClsHead, Accuracy
from .neck_gap import GlobalAveragePooling
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmdet.models.utils.positional_encoding import SineAgeEncoding
# from ..utils.implicit_semantic_data_aug import ISDALossCls
import torch.nn.functional as F
import numpy as np
import ipdb

print_tensor = lambda n, x: print(n, type(x), x.dtype, x.shape, x.min(), x.max())

@HEADS.register_module()
class LinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.

        age_encoding = dict(type='SineAgeEncoding', 
                            num_feats=256, normalize=True, max_age = 6),

    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 in_index = 0, 
                 dim = 3, dropout_ratio = 0.1,
                 topk=(1, ),
                 is_use_isda = False, 
                 isda_lambda = 2.5,
                 start_iters = 1,
                 max_iters = 4e5,
                 logit_dist_ratio = 0, 
                 confusion_loss_ratio = 0, 
                 age_encoding=dict(),
                 init_cfg = dict(type='TruncNormal', std = 0.2, layer=['Linear', 'Conv1d']), 
                 verb = False, **kwargs, 
                 ):
        super(LinearClsHead, self).__init__(loss=loss, topk=topk, init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.in_index = in_index
        self.dropout_ratio = dropout_ratio
        self.is_use_isda = is_use_isda
        self.isda_lambda = isda_lambda
        self._iter = start_iters
        self._max_iters = max_iters
        self.logit_dist_ratio = logit_dist_ratio
        self.confusion_loss_ratio = confusion_loss_ratio
        self.age_encoding_cfg = age_encoding
        # self.output_gap_feat1d = output_gap_feat1d
        self.use_sigmoid_cls = loss.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        
        self.verb = verb

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.gap = GlobalAveragePooling(dim = dim)
        self._init_layers()

        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout(self.dropout_ratio) 
        else:
            self.dropout = None
        
        self.compute_accuracy = Accuracy(topk=self.topk, 
                                        thresh=0.5 if self.use_sigmoid_cls else None, 
                                        use_sigmoid_act = self.use_sigmoid_cls)



    def _init_layers(self):
        self.fc = nn.Linear(self.in_channels, self.cls_out_channels)

        if self.age_encoding_cfg:
            self.age_encoding_cfg.pop('type', None)
            self.age_encoding  = SineAgeEncoding(**self.age_encoding_cfg)
            self.merge_layer = nn.Sequential(nn.Conv1d(self.in_channels, self.in_channels, kernel_size=2), 
                                             nn.GroupNorm(16, self.in_channels), 
                                             nn.ReLU(inplace = True))
            print('[AgeEncode]', self.age_encoding_cfg)
        else: 
            self.age_encoding = None            


    def forward_train(self, x, gt_label, age_step, train_cfg = None):

        if self.num_classes == 1 and not self.use_sigmoid_cls:
            gt_label = gt_label[:, 0]
        # gt_vector = gt_label.view(gt_label.shape[0], -1).max(-1).values

        ip = x[self.in_index] if isinstance(x, (tuple, list)) else x

        if self.verb: print_tensor(f'\n[ClsHead] gtcls {gt_label}; input ', ip)

        if self.dropout is not None: ip = self.dropout(ip) 

        # with torch.cuda.amp.autocast(enabled = False):
        gap_out = self.gap(ip.float()) # b1c > b2c? 

        if self.age_encoding:
            # print('Adding age embedding')
            age_embed = self.age_encoding(age_step)
            # print('raw gap', gap_out[0, : 8])
            # print('age embed', age_embed[0, :8])
            feat_merge = torch.stack([gap_out, age_embed], dim = -1) 
            # gap_out = gap_out + age_embed
            gap_out = self.merge_layer(feat_merge).squeeze()

            # print('raw gap time age', gap_out[0, : 8])
        # if self.verb: print_tensor('[ClsHead] post gap', gap_out)
        cls_score = self.fc(gap_out)

        if self.verb: 
            print_tensor('[ClsHead] gap', gap_out)
            print_tensor('[ClsHead] score', cls_score)
            for i in range(gap_out.shape[0]): print_tensor(f'[GAP] ix {i}', gap_out[i])

        losses = self.loss(cls_score, gt_label)

        neg_mask = gt_label[:, 0] == 0
        mild_mask = (gt_label[:, 0] != gt_label[:, 1]) # if two category have the same values
        severe_mask = gt_label[:, 1] == 1
        if self.logit_dist_ratio:
            c2c_distance = (neg_mask * 0 + severe_mask * 0 + mild_mask * 6).float() # nx2
            logit_distance = cls_score[:, 0] - cls_score[:, 1] 
            dist_loss = F.smooth_l1_loss(logit_distance, c2c_distance, 
                                        reduction= 'mean') * self.logit_dist_ratio
            # ipdb.set_trace()
            losses['loss'] = losses['loss'] + dist_loss 
            losses['dist_loss'] = dist_loss

        if self.confusion_loss_ratio:
            # adopted from "Pairwise Confusion for Fine-Grained Visual Classification", https://arxiv.org/abs/1705.08016 
            # 1xnxk, nx1xk > nxnxk
            sample_norm = torch.norm(cls_score[:, None, :] - cls_score[None, ...], dim = 2)
            sample_labels = gt_label.sum(dim = 1) # N, 
            sample_eq_mask = sample_labels[:, None] == sample_labels[None, :] # nx1, 1xn > nxn
            # ipdb.set_trace()
            confusion_loss = (sample_norm * sample_eq_mask).sum()/ sample_eq_mask.sum() * self.confusion_loss_ratio
            losses['loss'] = losses['loss'] + confusion_loss
            losses['EC_loss'] = confusion_loss

        return losses, gap_out
    
    def simple_test(self, x, age_step = None):
        """Test without augmentation.
            args: 
                x: feat_maps, multi-level
        """
        ip = x[self.in_index] if isinstance(x, (tuple, list)) else x
        if self.dropout is not None: ip = self.dropout(ip) 
        # print_tensor(f'[ClsHead] test input feat {self.in_index}', ip)
        gap_out = self.gap(ip)

        if age_step is not None and self.age_encoding is not None:
            age_embed = self.age_encoding(age_step)
            feat_merge = torch.stack([gap_out, age_embed], dim = -1) 
            gap_out = self.merge_layer(feat_merge).squeeze(dim = -1)
            # gap_out = gap_out + age_embed
        cls_score = self.fc(gap_out)
        return cls_score, gap_out 


@HEADS.register_module()
class LinearClsHeadCascade(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, ),
                 use_conv = True,
                 ):
        super(LinearClsHeadCascade, self).__init__(loss=loss, topk=topk)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_conv = use_conv

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')
        
        self.class_conv = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels, self.num_classes, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d(1)) if self.use_conv \
            else nn.Sequential(
                    GlobalAveragePooling(), 
                    nn.Linear(self.in_channels, self.num_classes)
            )

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, mean=0, std=0.01, bias=0)

    def forward(self, x, prev_outputs):
        """
        x: list of outputs of multi-level encoders
        prev_outputs: tuple(output_logit, output_dec)
        """
        ip = prev_outputs[-1] if isinstance(prev_outputs, (tuple, list)) else prev_outputs
        # print_tensor('res5', ip)
        # print_tensor('gt_class', gt_vector)
        # x = self.gap(ip) # 
        # cls_score = #self.fc(x)
        cls_score = self.class_conv(ip)
        return prev_outputs[0], cls_score

    def forward_train(self, x, prev_outputs, img_metas, gt_label, train_cfg):
        gt_vector = gt_label.view(gt_label.shape[0], -1).max(-1).values
        _, cls_score = self.forward(x, prev_outputs)

        cls_score = cls_score.view(cls_score.size(0), -1)
        # print_tensor('classpred', cls_score)
        # print_tensor('classtrue', gt_vector)
        losses = self.loss(cls_score, gt_vector)

        return losses
    
    def forward_test(self, inputs, prev_outputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return prev_outputs[0]