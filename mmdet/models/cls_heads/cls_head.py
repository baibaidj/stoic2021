import torch
import torch.nn.functional as F

from mmdet.models.losses import Accuracy
from ..builder import HEADS, build_loss
from .base_head import BaseHead
from mmcv.runner import force_fp32
import ipdb

@HEADS.register_module()
class ClsHead(BaseHead):
    """classification head.

    Args:
        loss (dict): Config of classification loss.
        topk (int | tuple): Top-k accuracy.
    """  # noqa: W605

    def __init__(self,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, )):
        super(ClsHead, self).__init__()

        assert isinstance(loss, dict)
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        self.compute_loss = build_loss(loss)
        self.compute_accuracy = Accuracy(topk=self.topk)

    @force_fp32(apply_to=('cls_score', ))
    def loss(self, cls_score, gt_label):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        with torch.cuda.amp.autocast(enabled = False):
            loss = self.compute_loss(cls_score.float(), gt_label.clone().detach(), avg_factor=num_samples)
        # compute accuracy
        with torch.no_grad():
            acc = self.compute_accuracy(cls_score, gt_label)
        # assert len(acc) == len(self.topk)
        losses['loss'] = loss

        for k, a in enumerate(acc): losses[f'acc_cls{k}'] = a
        # else:losses[f'acc_cls'] = acc
        # losses['accuracy'] = {f'top-{k}': a for k, a in zip(self.topk, acc)}
        return losses

    def forward_train(self, cls_score, gt_label):
        losses = self.loss(cls_score, gt_label)
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
