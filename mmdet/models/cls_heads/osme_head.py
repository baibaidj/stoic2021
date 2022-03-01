import torch
import torch.nn as nn
from mmcv.runner import BaseModule, force_fp32

from ..builder import HEADS
from .cls_head import ClsHead, Accuracy
from .neck_gap import GlobalAveragePooling
from mmdet.models.utils.positional_encoding import SineAgeEncoding
import torch.nn.functional as F
import numpy as np
import ipdb

print_tensor = lambda n, x: print(n, type(x), x.dtype, x.shape, x.min(), x.max())

@HEADS.register_module()
class OSMEClsHead(ClsHead):
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
                 logit_dist_ratio = 0, 
                 osme_cfg = dict(num_branch = 2, reduce_rate = 16, spatial_size = (8, 7, 7), 
                                 loss_weight = 0.5), 
                 age_encoding=dict(),
                 init_cfg = dict(type='TruncNormal', std = 0.2, layer=['Linear', 'Conv1d']),
                 verb = False, **kwargs, 
                 ):
        super(OSMEClsHead, self).__init__(loss=loss, topk=topk, init_cfg = init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.in_index = in_index
        self.dropout_ratio = dropout_ratio
        self.logit_dist_ratio = logit_dist_ratio

        self.mamc_loss_weight = osme_cfg.pop('loss_weight', 0)
        self.osme_cfg = osme_cfg
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
        self.osme_layer = OSMELayer(self.in_channels, **self.osme_cfg) 
        if self.mamc_loss_weight:
            self.mamc_loss = MAMCloss(self.osme_cfg.get('num_branch', 2))


    def forward_train(self, x, gt_label, age_step, train_cfg = None):

        if self.num_classes == 1 and not self.use_sigmoid_cls:
            gt_label = gt_label[:, 0]
        # gt_vector = gt_label.view(gt_label.shape[0], -1).max(-1).values

        feat_map_ip = x[self.in_index] if isinstance(x, (tuple, list)) else x

        if self.verb: print_tensor(f'\n[ClsHead] gtcls {gt_label}; input ', feat_map_ip)

        if self.dropout is not None: feat_map_ip = self.dropout(feat_map_ip) 

        # with torch.cuda.amp.autocast(enabled = False):
        gap_fc_init = self.gap(feat_map_ip) # b1c > b2c? 

        if self.age_encoding:
            # print('Adding age embedding')
            age_embed = self.age_encoding(age_step)
            feat_merge = torch.stack([gap_fc_init, age_embed], dim = -1) # BC2
            gap_fc_init = self.merge_layer(feat_merge).squeeze(dim = -1) # BC

        gap_fcs_bxpxc = self.osme_layer(feat_map_ip, gap_fc_init)
        
        gap_out = gap_fcs_bxpxc.sum(dim = 1)

        cls_score = self.fc(gap_out)

        if self.verb: 
            print_tensor('[ClsHead] gap', gap_out)
            print_tensor('[ClsHead] score', cls_score)
            for i in range(gap_out.shape[0]): print_tensor(f'[GAP] ix {i}', gap_out[i])


        # conventioanl sigmoid loss
        losses = self.loss(cls_score, gt_label)

        # mamc loss
        if self.mamc_loss_weight:
            mamc_loss = self.mamc_loss(gap_fcs_bxpxc, gt_label.sum(dim = 1)) * self.mamc_loss_weight
            # print(f'[MAMC] loss weight {self.mamc_loss_weight}', mamc_loss)
            losses['loss'] = losses['loss'] + mamc_loss
            losses['mamc_loss'] = mamc_loss

        # add logit distance loss
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

        return losses, gap_out
    
    def simple_test(self, x, age_step = None):
        """Test without augmentation.
            args: 
                x: feat_maps, multi-level
        """
        feat_map_ip = x[self.in_index] if isinstance(x, (tuple, list)) else x
        if self.dropout is not None: feat_map_ip = self.dropout(feat_map_ip) 

        # print_tensor(f'[ClsHead] test input feat {self.in_index}', feat_map_ip)
        gap_fc_init = self.gap(feat_map_ip) # b1c > b2c? 

        if self.age_encoding and age_step is not None:
            # print('Adding age embedding')
            age_embed = self.age_encoding(age_step)
            feat_merge = torch.stack([gap_fc_init, age_embed], dim = -1) 
            gap_fc_init = self.merge_layer(feat_merge).squeeze(dim = -1)

        gap_fcs_bxpxc = self.osme_layer(feat_map_ip, gap_fc_init)
        
        gap_out = gap_fcs_bxpxc.sum(dim = 1)

        cls_score = self.fc(gap_out)

        return cls_score, gap_out 


class OSMELayer(BaseModule):
    """
    implementation of one-squeeze multi-excitation module (OSME)  
    proposed in Multi-Attention Multi-Class Constraint for Fine-grained Image Recognition
    https://arxiv.org/abs/1806.05372
    
    """
    def __init__(self, num_feat, reduce_rate = 16, num_branch = 2, 
                spatial_size = (7, 5, 5), 
                init_cfg = dict(type='TruncNormal', std = 0.2, layer='Linear')):
        super().__init__(init_cfg = init_cfg)
        self.reduce_rate = reduce_rate
        self.num_branch = num_branch
        self.spatial_size = spatial_size
        self.spatial_vol = spatial_size[0] * spatial_size[1] * spatial_size[2]
        self.attention_creator = nn.ModuleList()
        self.attention_applier = nn.ModuleList()
        
        
        for i in range(num_branch):
            att_gen = nn.Sequential(
                        nn.Linear(num_feat, num_feat//reduce_rate), 
                        nn.ReLU(), 
                        nn.Linear(num_feat//reduce_rate, num_feat), 
                        nn.Sigmoid(),
            )
            self.attention_creator.append(att_gen)

            att_apply = nn.Sequential(
                    nn.Linear(num_feat * self.spatial_vol, num_feat), 
                    nn.LayerNorm(num_feat)
            )
            self.attention_applier.append(att_apply)

    def forward(self, feat_map, feat_fc):
        new_fcs = []
        # print_tensor('[OSME] map', feat_map)
        # print_tensor('[OSME] fc', feat_fc)
        for i, att_gen in enumerate(self.attention_creator):
            attention = att_gen(feat_fc)
            feat_map_new = feat_map * attention[:, :, None, None, None]
            # ipdb.set_trace()
            feat_fc_new = self.attention_applier[i](feat_map_new.flatten(1))
            # print_tensor('[OSME] att apply', feat_fc_new)
            new_fcs.append(feat_fc_new)
        new_fcs_bxpxc = torch.stack(new_fcs, dim = 1) # B, P, C
        return new_fcs_bxpxc
        
class MAMCloss(nn.Module):
    def __init__(self, num_branch) -> None:
        super(MAMCloss, self).__init__()
        self.num_branch = num_branch
        self.branch_eq_mask = torch.from_numpy(np.identity(num_branch)).to(torch.bool)


    @force_fp32(apply_to=('feat_fcs', 'targets'))
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
        B, P, C = feat_fcs.shape
        feat_fc_bpxc = feat_fcs.view(B * P, C)
        
        targets_bp = targets.repeat_interleave(P, 0).contiguous()
        

        class_eq_mask = targets_bp[:, None] == targets_bp[None, :] # bpx1, 1xbp > bpxbp
        branch_eq_mask = self.branch_eq_mask.to(targets.device
                                ).repeat_interleave(B, 0
                                ).repeat_interleave(B, 1) # bpxbp
        feat2feat_prod = feat_fc_bpxc @ feat_fc_bpxc.T
        feat2feat_prod = (feat2feat_prod - feat2feat_prod.mean()) / feat2feat_prod.std()

        loss_by_samples = []
        # print_tensor(f'[MAMC] feat fc', feat2feat_prod)
        for i in range(feat_fc_bpxc.shape[0]):
            sasc_mask = branch_eq_mask[i, :] & class_eq_mask[i, :]
            sadc_mask = branch_eq_mask[i, :] & ~class_eq_mask[i, :]
            dasc_mask = ~branch_eq_mask[i, :] & class_eq_mask[i, :]
            dadc_mask = ~branch_eq_mask[i, :] & ~class_eq_mask[i, :]
            
            # print(f'[MAMC] iter {i} feat sasc', sasc_mask)
            # print(f'[MAMC] iter {i} feat sadc', sadc_mask)
            # print(f'[MAMC] iter {i} feat dasc', dasc_mask)
            # print(f'[MAMC] iter {i} feat dadc', dadc_mask)
            
            pos_sasc = sasc_mask
            neg_sasc = sadc_mask | dasc_mask | dadc_mask
            sasc_loss = self.n_pair_loss(feat2feat_prod, pos_sasc, neg_sasc, anchor_ix=i)
            
            pos_sadc = sadc_mask
            neg_sadc = dadc_mask
            sadc_loss = self.n_pair_loss(feat2feat_prod, pos_sadc, neg_sadc, anchor_ix=i)

            pos_dasc = dasc_mask
            neg_dasc = dadc_mask
            dasc_loss = self.n_pair_loss(feat2feat_prod, pos_dasc, neg_dasc, anchor_ix=i)
            
            triloss = sasc_loss + sadc_loss + dasc_loss
            # print(f'[MAMC] iter {i}', sasc_loss, sadc_loss, dasc_loss, triloss)
            loss_by_samples.append(triloss)

        loss_mean = torch.stack(loss_by_samples, dim = 0).mean() / 3
        # print('[MAMCloss] actual', loss_mean)
        return loss_mean

    def n_pair_loss(self, feat2feat_prod, pos_mask, neg_mask, anchor_ix = 0):
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return 0

        neg2anc_n = feat2feat_prod[anchor_ix, neg_mask]
        pos2anc_p = feat2feat_prod[anchor_ix, pos_mask]
        neg2pos_nxp = neg2anc_n[:, None] - pos2anc_p[None, :]
        # if not norm, exponential will create inf
        neg2pos_nxp = (neg2pos_nxp - neg2pos_nxp.mean()) / neg2pos_nxp.std() 
        loss = torch.log(1 + torch.exp(neg2pos_nxp).sum(0)).sum()

        return loss
