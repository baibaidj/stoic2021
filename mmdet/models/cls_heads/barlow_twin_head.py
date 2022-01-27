import torch
import torch.nn.functional as F
from torch import nn
from .base_head import BaseHead
from ..builder import HEADS
from ..utils import print_tensor
from mmcv.runner import auto_fp16, force_fp32

@HEADS.register_module()
class BarlowTwinHead(BaseHead):
    """classification head.

    Args:
        
        lambd: weight on off-diagonal terms
        scale_loss: 'scale the loss'
    """  # noqa: W605

    def __init__(self,
                 mlp_channels = (128, 256, 512),  
                 in_index=0, sample_per_gpu = 2, num_gpus =1, 
                 scale_loss = 1/32, 
                 lambd = 3.9e-3,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 ):

        super(BarlowTwinHead, self).__init__()

        assert isinstance(loss, dict)
        self.num_gpus = num_gpus
        self.in_index = in_index
        self.batch_size_global = num_gpus * sample_per_gpu
        self.scale_loss = scale_loss
        self.lambd = lambd
        print('[BarlowHead] mlp_channels', mlp_channels)
        # projector
        sizes = mlp_channels#[2048] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward_train(self, feat1, feat2):
        xip1, xip2 = feat1[self.in_index], feat2[self.in_index]
        z1 = self.projector(xip1)
        z2 = self.projector(xip2)
        # print_tensor('z1', z1)
        # print_tensor('z2', z2)
        # empirical cross-correlation matrix
        cross_corr = self.bn(z1).T @ self.bn(z2)
        # sum the cross-correlation matrix between all gpus
        cross_corr.div_(self.batch_size_global)
        if self.num_gpus > 1: torch.distributed.all_reduce(cross_corr)
        losses = self.losses(cross_corr)
        return losses

    def simple_test(self, cls_score):
        """Test without augmentation."""
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred


    # @force_fp32()
    def losses(self, cross_corr):
        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        # print_tensor('cross corr', cross_corr)
        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum().mul(self.scale_loss)
        off_diag = off_diagonal(cross_corr).pow_(2).sum().mul(self.scale_loss)
        # print_tensor('on diag', on_diag)
        # print_tensor('off diag', off_diag)
        loss_ = on_diag + self.lambd * off_diag
        losses = {'loss': loss_}
        return losses



def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

