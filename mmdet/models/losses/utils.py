# Copyright (c) OpenMMLab. All rights reserved.
import functools

import mmcv, torch
import torch.nn.functional as F
from torch.autograd import Function, Variable

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


@mmcv.jit(derivate=True, coderize=True)
def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper

print_tensor = lambda n, x: print(n, type(x), x.dtype, x.shape, x.min(), x.max())

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
        X_in = X_in.long().view(num_element) # flatten X_in into a long vector
        out = Variable(self.ones.index_select(0, X_in)).view(output_size) # using label value as indexer to create one-hot
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).to(self.dtype)

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)




def group_onehot_prob(pred1hot : torch.Tensor, group_dict):
    """
    args:
        pred1hot: [tensor] in shape BC(D)HW, after softmax
        group_dict:  e.g. {0: [0], 1: [1,2,3], 2:[4,5]},  

    return 
        new_pred: [tensor] in shape  BG(D)HW, G=len(group_dict)
    """
    shape_origin = list(pred1hot.shape)
    shape_origin[1] = len(group_dict) # channel dim is replaced by the number of new groups
    new_pred = pred1hot.new_empty(shape_origin)
    for g, sub_classes in group_dict.items():
        new_pred[:, g, ...] = pred1hot[:, sub_classes, ...].sum(axis = 1)

    return new_pred

def group_class_decimal(gt, group_dict):
    """
    
    args:

    """
    new_pred = gt.new_empty(gt.shape)
    
    for g, sub_classes in group_dict.items():
        for c in sub_classes:
            new_pred[(gt == c).bool()] = g
    return new_pred