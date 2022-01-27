import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from .utils import *


def binary_ce_general(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index = None):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """

    if pred.dim() != label.dim():
        if pred.size(1) == 1: label = label.unsqueeze(1)
        else:label = One_Hot(pred.size(1))(label.contiguous())#_expand_onehot_labels(label, weight, pred.size(-1))

    # pdb.set_trace()
    # print_tensor('[BCE] pred', pred)
    # print_tensor('[BCE] label', label)
    B, C, *spatial_size = pred.shape
    bsc_order = [a+2 for a in range(len(spatial_size))]
    pred = pred.permute(0, *bsc_order, 1)
    label = label.permute(0, *bsc_order, 1)
    if ignore_index is not None: label[label == ignore_index] = 0
    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss_bce = F.binary_cross_entropy_with_logits(pred, label.float(), 
                pos_weight=class_weight, reduction='none')
    # fg_mask = label> 0
    # fg_pred = pred[fg_mask]
    # fg_label = label[fg_mask]
    # fg_loss = loss_bce[fg_mask]
    # loss = torch.mean(loss_bsc, dim = -1)
    # print_tensor(f'[BCE] non reduce loss', loss)
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss_bce, weight, reduction=reduction, avg_factor=avg_factor)

    return loss

@LOSSES.register_module()
class SoftDiceLoss(nn.Module):
    """
    a generalized class of dice loss where
    the ratio of false positive and false negative can be specified in the formula
    alpha stands for the contribution of false positive to the metric
    beta stands for the contribution of false negative to the metric

    To increase recall, one can increase beta to penalize false negative
    To increase precision, one can increase alpha

    """

    def __init__(self, num_classes, uncertain_map_alpha = 0, alpha = 1, beta = 1, 
                ignore_0 = True, is_gt1hot = False, verbose = False, centerline_dice_weight = 0,
                act = 'softmax'
                ):
        super(SoftDiceLoss, self).__init__()
        self.uncertain_map_alpha = uncertain_map_alpha
        self.alpha = alpha
        self.beta = beta 
        self.ignore_0 = ignore_0
        self.is_gt1hot = is_gt1hot
        self.verbose = verbose
        self.centerline_dice_weight = centerline_dice_weight
        if not is_gt1hot: self.update1hot_encoder(num_classes)

        if act == 'sigmoid': self.act = nn.Sigmoid()
        elif act == 'softmax': self.act = nn.Softmax(dim = 1)
        else: self.act = nn.Identity()

    def update1hot_encoder(self, num_classes):
        self.num_classes = num_classes
        self.one_hot_encoder = One_Hot(num_classes) #.forward

    def forward(self, pred, gt, weight = None, class_weight = None):
        """
        weight: spatial weight for each point/pixel
        """
        assert isinstance(class_weight, (torch.Tensor, type(None)))
        if class_weight is not None: 
            assert len(class_weight) == self.num_classes
            # if self.verbose: print('[DiceLoss] class weight', class_weight)

        smooth = 1.0 #1e-3
        batch_size = pred.size(0)     
        pred_prob = self.act(pred)
        gt_1hot = gt if self.is_gt1hot else self.one_hot_encoder(gt).contiguous()
        
        if self.verbose: 
            print_tensor('\n\t[DiceLoss] pred prob', pred_prob)
            print_tensor(f'\t[DiceLoss] gt1hot{gt_1hot.shape}, actual gt', gt)
        
        if self.centerline_dice_weight > 0: cl_dice_loss = self.cldice_loss(pred_prob, gt_1hot)

        pred_temp = pred_prob.view(batch_size, self.num_classes, -1)
        gt_1hot = gt_1hot.view(batch_size, self.num_classes, -1) #
        if weight is not None: 
            spatial_weight = weight.view(batch_size, 1, -1).expand(batch_size, self.num_classes, -1)
            # print_tensor('spatial weight', spatial_weight)
            pred_temp = pred_temp * spatial_weight
            gt_1hot = gt_1hot * spatial_weight

        if self.verbose:
            print_tensor('\t[Diceloss] pred', pred_temp)
            # print_tensor('[Diceloss] gt1hot', gt_1hot )
        # with GuruMeditation():
        if abs(self.alpha - 1) > 1e-5 or abs(self.beta - 1) > 1e-5: # TODO: this not right 
            pred_temp_minus = 1.0 - pred_temp
            gt_1hot_minus = 1.0 - gt_1hot
            tp = torch.sum(pred_temp * gt_1hot, 2) #+ smooth
            fp = torch.sum(pred_temp * gt_1hot_minus, 2)
            fn = torch.sum(pred_temp_minus * gt_1hot, 2) #(1.0- pred_temp) * (1.0 - gt_temp)
            dice_by_sample_class = (2.0 * tp + smooth) / ( self.alpha * fp + 2.0 * tp + self.beta * fn + smooth)
        # print_tensor('tp', tp); print_tensor('fp', fp); print_tensor('fn', fn)
        else:
            intersection = torch.sum(pred_temp * gt_1hot, 2) #+ smooth
            union = torch.sum(pred_temp, 2) + torch.sum(gt_1hot, 2)
            if self.verbose: 
                print_tensor('\t[Diceloss]intersection', intersection)
                print_tensor('\t[Diceloss]union' , union)
            # union = torch.pow(input, 2).sum(dim=axes) + torch.pow(target, 2).sum(dim=axes)
            # TODO: may generate RuntimeError: Function 'DivBackward0' returned nan values in its 0th output.
            dice_by_sample_class = (2.0 * intersection + smooth) / ( intersection + union + smooth)  #* (union == 0)
            if self.verbose: 
                print('\t\t[DiceLoss] by sample by class', dice_by_sample_class[0])

        mdice_by_class = torch.mean(dice_by_sample_class, dim = 0)
        start_class = 1 if (self.ignore_0 and self.num_classes > 1) else 0
        mdice_loss_by_class = 1.0 - mdice_by_class
        if self.verbose: print(f'\t[DiceLoss] mdiceloss final start{start_class}', mdice_loss_by_class)
        loss_ = torch.mean(mdice_loss_by_class[start_class:] * class_weight[start_class:], dim = 0)
        # print('nb_class %d;  start class %d' %(self.num_classes, start_class), mdice)
        if self.centerline_dice_weight > 0: loss_ = loss_ + cl_dice_loss
        if torch.isnan(loss_) : loss_ = pred_prob.sum() * 0.0
        return loss_


@LOSSES.register_module()
class RegressionLoss(nn.Module):
    def __init__(self, loss_weight = 1.0 , verbose = False, epislon = 1e-6):

        assert isinstance(loss_weight, (int, float))
        # self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.verbose = verbose
        self.epislon = epislon

        super(RegressionLoss, self).__init__()
        
    def forward(self, pred, gt, 
                    weight=None,
                    ignore_index=None,
                    ):
        assert pred.shape == gt.shape, f'pred {pred.shape} should has the same shape with gt {gt.shape}'
        out_dis = torch.sigmoid(pred) # SDF (-1, 1)
        # if gt.dim() < pred.dim():
        #     gt = self.one_hot_encoder(gt)#gt[:, None, ...]
        if self.verbose: print_tensor('RegLoss: pred', out_dis)
        if self.verbose: print_tensor('RegLoss: gt', gt)
        # this was adopted from TopNet https://link.springer.com/chapter/10.1007/978-3-030-59725-2_2
        # only compute the loss within the foreground region 
        # pdb.set_trace()
        loss_map = F.smooth_l1_loss(out_dis, gt, reduction = 'none')
        loss_map_weighted = loss_map / (torch.pow(gt, 2)  +  self.epislon) * (gt > 0)
        loss_dist = self.loss_weight * loss_map_weighted.mean()
        # if self.verbose: print_tensor('RegLoss: smoothl1', loss_map)
        # if self.verbose: print_tensor('RegLOss: gt_squared', gt_squared)
        # if self.verbose: print_tensor('RegLoss: mask', foreground_mask)
        # if self.verbose: print_tensor('RegLoss: loss_map_weighted', loss_map_weighted)
        return loss_dist
    

def get_uncertain_maps(pred_tensor, true_tensor, threshold = 0.37):
    """
    
    an implementation of the uncertain map proposed by 
        Zheng H, Chen Y, Yue X, et al. 
        Deep pancreas segmentation with uncertain regions of shadowed sets[J]. 
        Magnetic Resonance Imaging, 2020: 45-52.

    separation threshold: α split the prediction as three sets, certain foreground, certain background and uncertain region
    Prediction map: P
    Grounth truth map: Y
    uncertain weight map C 

    Ci = | Pi - Yi | (if α < Pi < 1- α) else 1

    args: 
        pred_tensor: prediciton probability in range(0, 1); post sigmoid or softmax!
        true_tensor: true label in range(0, 1)

    """
    certain_mask = (pred_tensor < threshold) & (pred_tensor > (1 - threshold))
    uncertain_mask = ~certain_mask
    uncertain_map = torch.abs(pred_tensor - true_tensor) * uncertain_mask + certain_mask
    return uncertain_map.long()


class CustomSoftDiceLoss(nn.Module):
    def __init__(self, num_classes, class_ids):
        super(CustomSoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(num_classes).forward
        self.num_classes = num_classes
        self.class_ids = class_ids

    def forward(self, pred, target):
        smooth = 0.01
        batch_size = pred.size(0)

        pred = F.softmax(pred[:,self.class_ids], dim=1).view(batch_size, len(self.class_ids), -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.num_classes, -1)
        target = target[:, self.class_ids, :]


        inter = torch.sum(pred * target, 2) + smooth
        union = torch.sum(pred, 2) + torch.sum(target, 2) + smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.num_classes))

        return score




def cross_entropy_2D(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


def cross_entropy_3D(input, target, weight=None, size_average=True):
    n, c, h, w, s = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss






if __name__ == '__main__':
    from torch.autograd import Variable
    depth=3
    batch_size=2
    encoder = One_Hot(depth=depth).forward
    y = Variable(torch.LongTensor(batch_size, 1, 1, 2 ,2).random_() % depth).cuda()  # 4 classes,1x3x3 img
    y_onehot = encoder(y)
    x = Variable(torch.randn(y_onehot.size()).float()).cuda()
    dicemetric = SoftDiceLoss(num_classes=depth)
    dicemetric(x,y)
