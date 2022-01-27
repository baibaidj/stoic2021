
import torch
from torch import nn
import torch.nn.functional as F
print_tensor = lambda n, x: print(n, type(x), x.dtype, x.shape, x.min(), x.max())

class EstimatorCV(nn.Module):
    def __init__(self, feature_num, class_num, device = None):
        super(EstimatorCV, self).__init__()

        self.class_num = class_num
        self.device = device
        self.CoVariance = torch.zeros(class_num, feature_num) #.to(device)
        self.Ave = torch.zeros(class_num, feature_num)# .to(device)
        self.Amount = torch.zeros(class_num)#.to(device)

    def update_CV(self, features, labels):
        """
        features_NHWxA.detach(), target_x_NHW

        torch.cuda.current_device()

        CoVarianece:
        
        Cov(A, B) = E(AB) - E(A)E(B) # E:expectation of 
        Cov(A, A) = E(A^2) - E(A)^2 
        """
        # device = features.device
        self.CoVariance = self.CoVariance.to(features.device)
        self.Ave = self.Ave.to(features.device)
        self.Amount = self.Amount.to(features.device)

        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )

        # transform the label value of background from 255 to 19 
        # label_mask = (labels == 255).long()
        # labels = ((1 - label_mask).mul(labels) + label_mask * 19).long()

        # the onehot version of ground truth
        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = var_temp.pow(2).sum(0).div(Amount_CxA)

        sum_weight_CV = onehot.sum(0).view(C, 1).expand(C, A)

        # import pdb; pdb.set_trace()
        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1).expand(C, A)
        )

        weight_CV[weight_CV != weight_CV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul((self.Ave - ave_CxA).pow(2))

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp.mul(weight_CV)
                          ).detach() + additional_CV.detach()

        self.Ave = (self.Ave.mul(1 - weight_CV) + ave_CxA.mul(weight_CV)).detach()

        self.Amount += onehot.sum(0)


class ISDALoss(nn.Module):

    """
    @inproceedings{NIPS2019_9426,
            title = {Implicit Semantic Data Augmentation for Deep Networks},
        author = {Wang, Yulin and Pan, Xuran and Song, Shiji and Zhang, Hong and Huang, Gao and Wu, Cheng},
        booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
            pages = {12635--12644},
            year = {2019},
    }

    @article{wang2021regularizing,
            title = {Regularizing deep networks with semantic data augmentation},
        author = {Wang, Yulin and Huang, Gao and Song, Shiji and Pan, Xuran and Xia, Yitong and Wu, Cheng},
        journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
            year = {2021}
    }

    adopted from 
    https://github.com/blackfeather-wang/ISDA-for-Deep-Networks
    https://zhuanlan.zhihu.com/p/344953635#ref_1

    """

    def __init__(self, feature_num, class_num, device = None, is_3d = False):
        super(ISDALoss, self).__init__()

        self.estimator = EstimatorCV(feature_num, class_num, device)
        self.is_3d = is_3d
        self.class_num = class_num

        # self.cross_entropy = nn.CrossEntropyLoss()

    def isda_aug(self, fc, features, y, labels, cv_matrix, ratio):

        """
        final_conv, features_NHWxA, y_NHWxC, target_x_NHW,
        self.estimator.CoVariance.detach(), ratio
        """
        label_mask = (labels == 255).long()
        labels = (1 - label_mask).mul(labels).long()

        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(fc.parameters())[0].squeeze()

        NxW_ij = weight_m.expand(N, C, A)

        NxW_kj = torch.gather(NxW_ij,
                              1,
                              labels.view(N, 1, 1)
                              .expand(N, C, A))

        CV_temp = cv_matrix[labels]

        sigma2 = ratio * (weight_m - NxW_kj).pow(2).mul(
            CV_temp.view(N, 1, A).expand(N, C, A)
        ).sum(2)

        aug_result = y + 0.5 * sigma2.mul((1 - label_mask).view(N, 1).expand(N, C))

        return aug_result

    def forward(self, *args, **kwargs):

        if self.is_3d:
            return self.forward_3d(*args, **kwargs)
        else:
            return self.forward_2d(*args, **kwargs)

    def forward_2d(self, features, final_conv, y, target_x, ratio):
        """
        feature: the feature maps of the second last layer, B, 512, HW
        final_conv: conv_seg, 
        y/preds: the logit outputs of the last layer, B, C, HW
        target_x: ground truth labels, B, HW
        ratio: args.lambda_0 * global_iteration / args.num_steps # training progress as percentage 
        """

        assert features.dim() == y.dim() == target_x.dim()
        N, A, H, W = features.size()
        # target_x = target_x.view(N, 1, target_x.size(1), target_x.size(2)).float() #
        target_x = F.interpolate(target_x.float(), size=(H, W), mode='nearest', align_corners=None)

        target_x = target_x.long().squeeze()

        C = self.class_num

        features_NHWxA = features.permute(0, 2, 3, 1).contiguous().view(N * H * W, A)

        target_x_NHW = target_x.contiguous().view(N * H * W)

        y_NHWxC = y.permute(0, 2, 3, 1).contiguous().view(N * H * W, C)

        self.estimator.update_CV(features_NHWxA.detach(), target_x_NHW)

        isda_aug_y_NHWxC = self.isda_aug(final_conv, features_NHWxA, y_NHWxC, target_x_NHW,
                                         self.estimator.CoVariance.detach(), ratio)

        isda_aug_y = isda_aug_y_NHWxC.view(N, H, W, C).permute(0, 3, 1, 2)

        return isda_aug_y

    def forward_3d(self, features, final_conv, y, target_x, ratio):
        """
        feature: the feature maps of the second last layer, B, 512, HW
        final_conv: conv_seg, 
        y/preds: the logit outputs of the last layer, B, C, HW
        target_x: ground truth labels, B, HW
        ratio: args.lambda_0 * global_iteration / args.num_steps # training progress as percentage 
        """

        assert features.dim() == y.dim() == target_x.dim()
        N, A, T, H, W = features.size()
        # target_x = target_x.view(N, 1, target_x.size(1), target_x.size(2)).float() #
        target_x = F.interpolate(target_x.float(), size=(T, H, W), mode='nearest', align_corners=None)

        target_x = target_x.long().squeeze()

        C = self.class_num
        # NATHW > NTHWA
        features_NHWxA = features.permute(0, 2, 3, 4, 1).contiguous().view(N * T * H * W, A)

        target_x_NHW = target_x.contiguous().view(N * T * H * W)

        y_NHWxC = y.permute(0, 2, 3, 4, 1).contiguous().view(N * T * H * W, C)

        self.estimator.update_CV(features_NHWxA.detach(), target_x_NHW)

        isda_aug_y_NHWxC = self.isda_aug(final_conv, features_NHWxA, y_NHWxC, target_x_NHW,
                                         self.estimator.CoVariance.detach(), ratio)

        isda_aug_y = isda_aug_y_NHWxC.view(N, T, H, W, C).permute(0, 4, 1, 2, 3)

        return isda_aug_y





class ISDALossCls(nn.Module):

    """
    @inproceedings{NIPS2019_9426,
            title = {Implicit Semantic Data Augmentation for Deep Networks},
        author = {Wang, Yulin and Pan, Xuran and Song, Shiji and Zhang, Hong and Huang, Gao and Wu, Cheng},
        booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
            pages = {12635--12644},
            year = {2019},
    }

    @article{wang2021regularizing,
            title = {Regularizing deep networks with semantic data augmentation},
        author = {Wang, Yulin and Huang, Gao and Song, Shiji and Pan, Xuran and Xia, Yitong and Wu, Cheng},
        journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
            year = {2021}
    }

    adopted from 
    https://github.com/blackfeather-wang/ISDA-for-Deep-Networks
    https://zhuanlan.zhihu.com/p/344953635#ref_1

    """

    def __init__(self, feature_num, class_num, device = None):
        super(ISDALossCls, self).__init__()

        self.estimator = EstimatorCV(feature_num, class_num, device)
        self.class_num = class_num

        # self.cross_entropy = nn.CrossEntropyLoss()

    def isda_aug(self, fc, features, y, labels, cv_matrix, ratio):

        """
        final_conv, features_NHWxA, y_NHWxC, target_x_NHW,
        self.estimator.CoVariance.detach(), ratio
        """

        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(fc.parameters())[0]

        NxW_ij = weight_m.expand(N, C, A)

        NxW_kj = torch.gather(NxW_ij,
                              1,
                              labels.view(N, 1, 1)
                              .expand(N, C, A))

        CV_temp = cv_matrix[labels]

        sigma2 = ratio * (weight_m - NxW_kj).pow(2).mul(
            CV_temp.view(N, 1, A).expand(N, C, A)
        ).sum(2)

        aug_result = y + 0.5 * sigma2

        return aug_result


    def forward(self, features, final_conv, pred, target_x, ratio):
        """
        feature: the feature maps of the second last layer, B, Cf
        final_conv: fully-connected , 
        y/preds: the logit outputs of the last layer,  B, Cc
        target_x: ground truth labels, B
        ratio: args.lambda_0 * global_iteration / args.num_steps # training progress as percentage 
        """
        # print_tensor('feat', features)
        # print_tensor('pred', pred)
        # print_tensor('target', target_x)
        assert features.shape[0] == pred.shape[0] == target_x.shape[0]

        self.estimator.update_CV(features.detach(), target_x)

        isda_aug_y = self.isda_aug(final_conv, features, pred, target_x,
                                    self.estimator.CoVariance.detach(), ratio)

        # print_tensor('aug pred', isda_aug_y)
        return isda_aug_y
