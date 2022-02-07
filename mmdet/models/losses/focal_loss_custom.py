
from .focal_loss import (
    torch, nn, F, LOSSES, weight_reduce_loss,
    print_tensor, )
import ipdb


def focal_loss_with_prob_multitask(pred,
                            target,
                            weight=None,
                            class_weight = None, 
                            gamma=2.0,
                            alpha=0.25,
                            reduction='mean',
                            avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Different from `py_sigmoid_focal_loss`, this function accepts probability
    as input.

    Args:
        pred (torch.Tensor): The prediction probability with shape (N, C),
            C is the number of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # assert target.shape == pred.shape[:2]

    # print_tensor('[FocalLoss] pred', pred)
    # print_tensor('[FocalLoss] gt', target)
    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss_bce = F.binary_cross_entropy(pred, target, reduction='none') 
    loss = loss_bce * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    # ipdb.set_trace()
    if class_weight is not None:
        loss = loss * class_weight[None, :]
    
    loss_by_class = loss.mean(dim = 0)
    loss = loss_by_class.sum()

    # loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    # print('prob', pred)
    # print('target', target)
    # print('loss_bce\n', loss_bce)
    # print('loss_weight\n', focal_weight)
    # print(f'[SumLoss] by class {loss_by_class} final {loss} ')

    # ipdb.set_trace()
    return loss



@LOSSES.register_module()
class FocalLossMultitask(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 class_weight = None, 
                 loss_weight=1.0, 
                 perform_act=True, 
                 verbose = False):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            activated (bool, optional): Whether the input is activated.
                If True, it means the input has been activated and can be
                treated as probabilities. Else, it should be treated as logits.
                Defaults to False.
        """
        super(FocalLossMultitask, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.perform_act = perform_act
        self.class_weight = class_weight
        self.verbose = verbose

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        NOTE: expect the target without channels, and will be transformed to 1hot (0 will occupy a channel)
        So, this loss function does not support 1 channel prediction 
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        if isinstance(self.class_weight, (tuple, list)):
            self.class_weight = pred.new_tensor(self.class_weight)

        if self.use_sigmoid:
            
            if self.perform_act:
                pred = pred.sigmoid()

            calculate_loss_func = focal_loss_with_prob_multitask
            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                class_weight = self.class_weight, 
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

            if self.verbose: 
                torch.set_printoptions(precision=2)
                count_mask = weight>0
                count_target = target[count_mask]
                count_pred = pred[count_mask]
                # pred1hot = torch.cat([torch.ones_like(pred[fg_mask]) , pred[fg_mask] ], axis = 1)
                fg_loss = calculate_loss_func(count_pred, count_target, reduction='none')
                fg_pred_nxc = torch.cat([count_pred, torch.sigmoid(count_pred), count_target[:, None], fg_loss], axis = 1)
                # fg_counts, weight_counts = target.sum(), weight.sum()
                print_tensor('[Focalloss] target cls', target)
                print(f'[Focalloss] fg logit gt loss \n {fg_pred_nxc[:16]}', fg_pred_nxc.shape)
                ipdb.set_trace()
                # counts fg-{fg_counts} weight-{weight_counts},
                # print_tensor(f'[FocalLoss] pred', pred )
                # print_tensor(f'[Focalloss] gt', target)
                # if weight is not None:
                #     print_tensor(f'[Focalloss] weight', weight)
                # print_tensor(f'[Focalloss] loss', loss_cls)
        else:
            raise NotImplementedError
        return loss_cls