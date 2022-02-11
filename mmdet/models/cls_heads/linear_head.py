import torch
import torch.nn as nn
from mmcv.cnn import normal_init, kaiming_init, constant_init

from ..builder import HEADS
from .cls_head import ClsHead, Accuracy
from .neck_gap import GlobalAveragePooling
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmdet.models.utils.positional_encoding import SineAgeEncoding
# from ..utils.implicit_semantic_data_aug import ISDALossCls
import torch.nn.functional as F
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
                 add_feat_dist = False, 
                 age_encoding=dict(),
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
        self.logit_dist_ratio = logit_dist_ratio
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

        if age_encoding:
            age_encoding.pop('type', None)
            self.age_encoding  = SineAgeEncoding(**age_encoding)
            print('[AgeEncode]', age_encoding)
        else: 
            self.age_encoding = None


    def _init_layers(self):
        # if self.logit_dist_ratio:
        #     print('[FeatDistance] for two coorelated classes@@')
        #     self.fc4cls0 = nn.ModuleList([
        #                         nn.Sequential(
        #                             nn.Linear(self.in_channels, self.in_channels//4), 
        #                             nn.LayerNorm(self.in_channels //4), 
        #                             nn.GELU()), 
        #                     nn.Linear(self.in_channels//4, 1)])
        #     # self
        #     self.fc4cls1 = nn.ModuleList([
        #                         nn.Sequential(
        #                             nn.Linear(self.in_channels, self.in_channels//4), 
        #                             nn.LayerNorm(self.in_channels //4), 
        #                             nn.GELU()), 
        #                     nn.Linear(self.in_channels//4, 1)])
        
        # else:
        self.fc = nn.Linear(self.in_channels, self.cls_out_channels)


    def init_weights(self):
        # if self.logit_dist_ratio:
        #     for m in self.fc4cls0:
        #         if isinstance(m, nn.Linear):
        #             normal_init(m, mean=0, std=0.01, bias=0)
        #     for m in self.fc4cls1:
        #         if isinstance(m, nn.Linear):
        #             normal_init(m, mean=0, std=0.01, bias=0)
        # else:
        normal_init(self.fc, mean=0, std=0.02, bias=0)


    def forward_train(self, x, gt_label, age_step, train_cfg = None):

        if self.num_classes == 1 and not self.use_sigmoid_cls:
            gt_label = gt_label[:, 0]
        # gt_vector = gt_label.view(gt_label.shape[0], -1).max(-1).values

        ip = x[self.in_index] if isinstance(x, (tuple, list)) else x

        if self.verb: print_tensor(f'[ClsHead] gtcls {gt_label}; input ', ip)

        if self.dropout is not None: ip = self.dropout(ip) 

        # with torch.cuda.amp.autocast(enabled = False):
        gap_out = self.gap(ip.float()) # b1c > b2c? 

        if self.age_encoding:
            # print('Adding age embedding')
            age_embed = self.age_encoding(age_step)
            # print('raw gap', gap_out[0, : 8])
            # print('age embed', age_embed[0, :8])
            gap_out = gap_out + age_embed
            # print('raw gap time age', gap_out[0, : 8])
        # if self.verb: print_tensor('[ClsHead] post gap', gap_out)

        # ipdb.set_trace()
            # print('Separate feature vectors for each class')
            # feat4cls0 = self.fc4cls0[0](gap_out) # bc
            # feat4cls1 = self.fc4cls1[0](gap_out) # bc

            # cls_score0 = self.fc4cls0[1](feat4cls0)
            # cls_score1 = self.fc4cls1[1](feat4cls1)
            # cls_score = torch.cat([cls_score0, cls_score1], dim = 1)

            # feat_distance = torch.linalg.norm(feat4cls0 - feat4cls1, 2, dim = 1)
            # dist_loss = F.smooth_l1_loss(feat_distance, c2c_distance, reduction= 'mean') * 0.3
        # else:
        cls_score = self.fc(gap_out)

        if self.verb: print_tensor('[ClsHead] score', cls_score)
        # ipdb.set_trace()
        losses = self.loss(cls_score, gt_label)

        if self.logit_dist_ratio:
            # logit distance
            neg_mask = gt_label[:, 0] == 0
            mild_mask = (gt_label[:, 0] == 1) * (gt_label[:, 1] == 0) # if two category have the same values
            severe_mask = gt_label[:, 1] == 1
            c2c_distance = (neg_mask * 0 + severe_mask * 0 + mild_mask * 6).float()
            logit_distance = torch.abs(cls_score[:, 0] - cls_score[:, 1])
            dist_loss = F.smooth_l1_loss(logit_distance, c2c_distance, 
                                        reduction= 'mean') * self.logit_dist_ratio

            losses['loss'] = losses['loss'] + dist_loss 
            losses['dist_loss'] = dist_loss

        return losses, gap_out
    
    def simple_test(self, x, age_step = None):
        """Test without augmentation.
            args: 
                x: feat_maps, multi-level
        """
        ip = x[self.in_index] if isinstance(x, (tuple, list)) else x

        # print_tensor(f'[ClsHead] test input feat {self.in_index}', ip)
        gap_out = self.gap(ip)

        if age_step is not None and self.age_encoding is not None:
            age_embed = self.age_encoding(age_step)
            gap_out = gap_out + age_embed

        # if self.logit_dist_ratio:
        #     feat4cls0 = self.fc4cls0[0](gap_out) # bc
        #     feat4cls1 = self.fc4cls1[0](gap_out) # bc

        #     cls_score0 = self.fc4cls0[1](feat4cls0)
        #     cls_score1 = self.fc4cls1[1](feat4cls1)
        #     cls_score = torch.cat([cls_score0, cls_score1], dim = 1)
        # else:
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