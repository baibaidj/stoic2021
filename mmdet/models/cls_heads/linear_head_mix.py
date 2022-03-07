import torch
import torch.nn as nn

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
class LinearClsMixHead(ClsHead):
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
                 confusion_loss_ratio = 0, 
                 demo_encode_cfg=dict(age_category = 6, gender_category = 3, image_feat_num = 2),
                 init_cfg = dict(type='TruncNormal', std = 0.2, layer=['Linear', 'Conv1d']), 
                 verb = False, **kwargs, 
                 ):
        super(LinearClsMixHead, self).__init__(loss=loss, topk=topk, init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.in_index = in_index
        self.dropout_ratio = dropout_ratio
        self.logit_dist_ratio = logit_dist_ratio
        self.confusion_loss_ratio = confusion_loss_ratio
        print('[DemographicEncode]', demo_encode_cfg)
        self.age_category  = demo_encode_cfg.pop('age_category', None)
        self.gender_category = demo_encode_cfg.pop('gender_category', None)
        self.image_feat_num = demo_encode_cfg.pop('image_feat_num', None)
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

        if self.age_category:
            self.image_feat_reducer = nn.Linear(self.in_channels, self.image_feat_num)
            self.num_feat_combine = self.age_category + self.gender_category + self.image_feat_num
            self.fc = nn.Linear(self.num_feat_combine, self.cls_out_channels)
            self.age1hot_func = One_Hot(self.age_category)
            self.gender1hot_func = One_Hot(self.gender_category)
        else: 
            self.fc = nn.Linear(self.in_channels, self.cls_out_channels)         


    def forward_train(self, x, gt_label, age_step, gender_step, train_cfg = None):

        if self.num_classes == 1 and not self.use_sigmoid_cls:
            gt_label = gt_label[:, 0]
        # gt_vector = gt_label.view(gt_label.shape[0], -1).max(-1).values

        ip = x[self.in_index] if isinstance(x, (tuple, list)) else x

        if self.verb: print_tensor(f'\n[ClsHead] gtcls {gt_label}; input ', ip)

        if self.dropout is not None: ip = self.dropout(ip) 

        # with torch.cuda.amp.autocast(enabled = False):
        gap_out = self.gap(ip) # b1c > b2c? 

        if self.age_category:
            age1hot = self.age1hot_func(age_step - 1) # bxc, age start from 1
            gender1hot = self.gender1hot_func(gender_step) # bxc, gender start from 0
            image_feat_distill = self.image_feat_reducer(gap_out)
            feat_combine = torch.cat([image_feat_distill, age1hot, gender1hot], dim = 1)
            cls_score = self.fc(feat_combine)
        else:
            cls_score = self.fc(gap_out)
        # if self.verb: print_tensor('[ClsHead] post gap', gap_out)
    
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
    
    def simple_test(self, x, age_step = None, gender_step = None):
        """Test without augmentation.
            args: 
                x: feat_maps, multi-level
        """
        # print_tensor('Test age', age_step )
        # print_tensor('Test gender', gender_step)
        ip = x[self.in_index] if isinstance(x, (tuple, list)) else x
        if self.dropout is not None: ip = self.dropout(ip) 
        # print_tensor(f'[ClsHead] test input feat {self.in_index}', ip)
        gap_out = self.gap(ip)

        if self.age_category:
            age1hot = self.age1hot_func(age_step - 1) # bxc
            gender1hot = self.gender1hot_func(gender_step) # bxc
            image_feat_distill = self.image_feat_reducer(gap_out)
            feat_combine = torch.cat([image_feat_distill, age1hot, gender1hot], dim = 1)
            cls_score = self.fc(feat_combine)
        else:
            # gap_out = gap_out + age_embed
            cls_score = self.fc(gap_out)
        return cls_score, gap_out 

from torch.autograd import Variable
class One_Hot(object):
    """transform the value in mask into one-hot representation
        depth: number of unique value in the mask
    """
    def __init__(self, depth, dtype = torch.float):
        # super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth) #.cuda() # identity matrix, diagnal is 1 
        self.dtype = dtype

    def __call__(self, X_in : torch.Tensor):
        # print_tensor('Ones', self.ones)
        self.ones = self.ones.to(X_in.device)
        if self.depth <= 1:
            return X_in.unsqueeze(1)

        n_dim = X_in.dim()
        output_size = X_in.size() + torch.Size([self.depth])
        # print_tensor(f'[1Hot] output size:{output_size} ; input:', X_in)
        num_element = X_in.numel()
        X_in = X_in.data.long().view(num_element) # flatten X_in into a long vector
        out = Variable(self.ones.index_select(0, X_in)).view(output_size) # using label value as indexer to create one-hot
        if out.dim() == 2:
            return out.to(self.dtype)
        else:
            return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).to(self.dtype)

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)