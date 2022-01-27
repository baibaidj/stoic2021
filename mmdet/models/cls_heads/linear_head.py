import torch
import torch.nn as nn
from mmcv.cnn import normal_init, kaiming_init, constant_init

from ..builder import HEADS
from .cls_head import ClsHead, Accuracy
from .neck_gap import GlobalAveragePooling
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..utils.implicit_semantic_data_aug import ISDALossCls
import pdb

print_tensor = lambda n, x: print(n, type(x), x.dtype, x.shape, x.min(), x.max())

@HEADS.register_module()
class LinearClsHead(ClsHead):
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
                 in_index = 0, 
                 dim = 3, dropout_ratio = 0.1,
                 topk=(1, ),
                 is_use_isda = False, 
                 isda_lambda = 2.5,
                 start_iters = 1,
                 max_iters = 4e5,
                 verb = False
                 ):
        super(LinearClsHead, self).__init__(loss=loss, topk=topk)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.in_index = in_index
        self.dropout_ratio = dropout_ratio
        self.is_use_isda = is_use_isda
        self.isda_lambda = isda_lambda
        self._iter = start_iters
        self._max_iters = max_iters
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

        # TODO build this loss for classification
        if is_use_isda:
            self.isda_augmentor = ISDALossCls(self.in_channels, self.cls_out_channels)
        
        self.compute_accuracy = Accuracy(topk=self.topk, thresh=0.5, is_multi_task = True)

    def _init_layers(self):
        self.fc = nn.Linear(self.in_channels, self.cls_out_channels)

    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)

    def forward_train(self, x, gt_label, train_cfg = None):
        # gt_vector = gt_label.view(gt_label.shape[0], -1).max(-1).values
        gt_vector = gt_label
        ip = x[self.in_index] if isinstance(x, (tuple, list)) else x

        if self.verb: print_tensor(f'[ClsHead] gtcls {gt_label}; input ', ip)

        if self.dropout is not None: ip = self.dropout(ip) 
        ip_dtype = ip.dtype
        with torch.cuda.amp.autocast(enabled = False):
            gap_out = self.gap(ip.float())

        if self.verb: print_tensor('[ClsHead] post gap', gap_out)
        cls_score = self.fc(gap_out)
        if self.verb: print_tensor('[ClsHead] score', cls_score)

        if self.is_use_isda:
            ratio = min(self.isda_lambda * self._iter, self._max_iters) / self._max_iters
            cls_score = self.isda_augmentor(x.detach(), self.fc, cls_score, gt_label, ratio)
            self._iter += 1

        losses = self.loss(cls_score, gt_vector)
        return losses, gap_out
    
    def simple_test(self, x):
        """Test without augmentation.
            args: 
                x: feat_maps, multi-level
        """
        ip = x[self.in_index] if isinstance(x, (tuple, list)) else x
        gap_out = self.gap(ip)
        cls_score = self.fc(gap_out)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = cls_score
        if torch.onnx.is_in_onnx_export():
            return pred
        # pred = list(pred.detach().cpu().numpy())
        return pred, gap_out



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