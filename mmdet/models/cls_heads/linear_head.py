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
                 verb = False, **kwargs, 
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

        if self.age_encoding_cfg:
            self.age_encoding_cfg.pop('type', None)
            self.age_encoding  = SineAgeEncoding(**self.age_encoding_cfg)
            self.merge_layer = nn.Sequential(nn.Conv1d(self.in_channels, self.in_channels, kernel_size=2), 
                                             nn.GroupNorm(16, self.in_channels), 
                                             nn.ReLU(inplace = True))
            print('[AgeEncode]', self.age_encoding_cfg)
        else: 
            self.age_encoding = None            

    def init_weights(self):
        # if self.logit_dist_ratio:
        #     for m in self.fc4cls0:
        #         if isinstance(m, nn.Linear):
        #             normal_init(m, mean=0, std=0.01, bias=0)
        #     for m in self.fc4cls1:
        #         if isinstance(m, nn.Linear):
        #             normal_init(m, mean=0, std=0.01, bias=0)
        # else:
        normal_init(self.fc, mean=0, std=0.2, bias=0)
        if self.age_encoding:
            trunc_normal_init(self.merge_layer[0])


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

        # print_tensor(f'[ClsHead] test input feat {self.in_index}', ip)
        gap_out = self.gap(ip)

        if age_step is not None and self.age_encoding is not None:
            age_embed = self.age_encoding(age_step)
            feat_merge = torch.stack([gap_out, age_embed], dim = -1) 
            gap_out = self.merge_layer(feat_merge).squeeze()
            # gap_out = gap_out + age_embed

        # if self.logit_dist_ratio:
        #     feat4cls0 = self.fc4cls0[0](gap_out) # bc
        #     feat4cls1 = self.fc4cls1[0](gap_out) # bc

        #     cls_score0 = self.fc4cls0[1](feat4cls0)
        #     cls_score1 = self.fc4cls1[1](feat4cls1)
        #     cls_score = torch.cat([cls_score0, cls_score1], dim = 1)
        # else:
        cls_score = self.fc(gap_out)
        # if isinstance(cls_score, list):
        #     cls_score = sum(cls_score) / float(len(cls_score))
        # pred = cls_score
        # if torch.onnx.is_in_onnx_export():
        #     return pred
        # pred = list(pred.detach().cpu().numpy())
        return cls_score, gap_out 


class OSMELayer(BaseModule):
    """
    implementation of one-squeeze multi-excitation module (OSME)  
    proposed in Multi-Attention Multi-Class Constraint for Fine-grained Image Recognition
    https://arxiv.org/abs/1806.05372
    
    """
    def __init__(self, num_feat, reduce_rate = 16, num_branch = 2, 
                spatial_size = (7, 5, 5), 
                init_cfg = None):
        super().__init__(init_cfg = init_cfg)
        self.reduce_rate = reduce_rate
        self.num_branch = num_branch
        self.spatial_size = spatial_size
        self.spatial_vol = spatial_size[0] * spatial_size[1] * spatial_size[2]
        self.attention_creator = nn.ModuleList()
        self.attention_applier = nn.ModuleList()
        
        
        for i in range(num_branch):
            att_gen = nn.Sequential(
                        nn.Linear(num_feat, num_feat/reduce_rate), 
                        nn.ReLU(), 
                        nn.Linear(num_feat/reduce_rate, num_feat), 
                        nn.Sigmoid(),
            )
            self.attention_creator.append(att_gen)

            att_apply = nn.Linear(num_feat * self.spatial_vol, num_feat)
            self.attention_applier.append(att_apply)

    def forward(self, feat_map, feat_fc):
        new_fcs = []
        for i, att_gen in enumerate(self.attention_creator):
            attention = att_gen(feat_fc)
            feat_map_new = feat_map * attention
            feat_fc_new = self.attention_applier[i](feat_map_new.flatten(2))
            new_fcs.append(feat_fc_new)
        return new_fcs
        

class MAMCloss(object):
    def __init__(self, num_branch) -> None:
        self.num_branch = num_branch

        self.branch_eq_mask = torch.from_numpy(np.identity(num_branch))
        pass

    def forward(self, feat_fcs, targets):
        """
        feat_fcs, P, B, C, 
        num_fc: P * B

        for every fc, we need to find the related positives and negatives 
        3 categories: 
        1. same attention and same class
        2. same attention and different class
        3. different attention and same class
        4. different attention and different class

        2 x 16 = 32 
        32 x 32
        Args:
            feat_fcs: list([fc_bxc])
            targets: bx1
        """
        feat_fcs = torch.stack(feat_fcs, dim = 0)
        B, P, C = feat_fcs.shape
        feat_fc_bpxc = feat_fcs.view(B * P, C)
        
        targets_bp = targets.repeat_interleave(P, 0).contiguous()
        

        class_eq_mask = targets_bp[:, None] == targets_bp[None, :] # bpx1, 1xbp > bpxbp
        branch_eq_mask = self.branch_eq_mask.to(targets.device
                                ).repeat_interleave(B, 0
                                ).repeat_interleave(B, 1) # bpxbp

        loss_by_samples = []
        for i in range(feat_fc_bpxc.shape[0]):
            fc_i = feat_fc_bpxc[i]
            sasc_mask = branch_eq_mask[i, :] & class_eq_mask[i, :]
            sadc_mask = branch_eq_mask[i, :] & ~class_eq_mask[i, :]
            dasc_mask = ~branch_eq_mask[i, :] & class_eq_mask[i, :]
            dadc_mask = ~branch_eq_mask[i, :] & ~class_eq_mask[i, :]
            
            pos_sasc = sasc_mask
            neg_sasc = sadc_mask | dasc_mask | dadc_mask
            sasc_loss = self.n_pair_loss(fc_i, feat_fc_bpxc, pos_sasc, neg_sasc)
            
            pos_sadc = sadc_mask
            neg_sadc = dadc_mask
            sadc_loss = self.n_pair_loss(fc_i, feat_fc_bpxc, pos_sadc, neg_sadc)

            pos_dasc = dasc_mask
            neg_dasc = dadc_mask
            dasc_loss = self.n_pair_loss(fc_i, feat_fc_bpxc, pos_dasc, neg_dasc)
            
            triloss = sasc_loss + sadc_loss + dasc_loss
            loss_by_samples.append(triloss)
        
        return torch.mean(loss_by_samples)

    def n_pair_loss(self, fc_anchor, fc_bpxc, pos_mask, neg_mask):

        pos_fcs_pxc = fc_bpxc[pos_mask]
        neg_fcs_nxc = fc_bpxc[neg_mask]
 
        neg2anc_nx1 = torch.dot(neg_fcs_nxc, fc_anchor[:, None])  # 
        pos2anc_px1 = torch.dot(pos_fcs_pxc, fc_anchor[:, None]) 
        neg2pos_nxp = torch.exp(neg2anc_nx1[:, None] - pos2anc_px1[None, :])
        loss = torch.log(1 + neg2pos_nxp.sum(0)).sum()

        return loss


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