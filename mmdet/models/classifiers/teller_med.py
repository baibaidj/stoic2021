import torch

from ..builder import DETECTORS
# from ..utils import BatchMixupLayer
from .image import ImageClassifier
from mmdet.datasets.pipelines.compose import Compose
from mmdet.utils.resize import list_dict2dict_list
from ..utils import print_tensor
# from torch.profiler import profile, record_function, ProfilerActivity
import ipdb


@DETECTORS.register_module()
class ImageClassifierMed(ImageClassifier):

    def __init__(self, *args, 
                 gpu_aug_pipelines = None, 
                 **kwargs
                 ):
        super(ImageClassifierMed, self).__init__(*args, **kwargs)
        # ipdb.set_trace()
        self.target_class = self.test_cfg.get('target_class', None)
        self.use_sigmoid = self.head.use_sigmoid_cls
        self.gpu_pipelines = Compose(gpu_aug_pipelines) if gpu_aug_pipelines is not None else None

    @torch.no_grad()
    def update_img_metas(self, imgs, img_metas, **kwargs):
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        # TODO: adjust keys
        # gt_keys = ['img'] # 'img_metas'
        data_dict = list_dict2dict_list(img_metas, verbose=False)
        data_dict.update({'img': imgs}) # , 'seg': seg
        data_dict = self.gpu_pipelines(data_dict)
        # for b, m in enumerate(img_metas): m['patch_shape'] = data_dict['patch_shape']

        cls_gt_list = [m['img_meta_dict']['target_class'] for m in img_metas]
        gt_label = torch.tensor(cls_gt_list, dtype = torch.long, device=imgs.device) 
        if isinstance(self.target_class, int):
            gt_label = gt_label[:, self.target_class : self.target_class + 1]
        ages = torch.tensor([m['img_meta_dict']['age'] for m in img_metas], 
                            dtype = imgs.dtype, device = imgs.device)
        filenames = [m['img_meta_dict']['filename_or_obj'].split('/')[-1] for m in img_metas]
        
        # for i, fn in enumerate(filenames): 
        #     print(f'{fn} : age {ages[i]} covid+severe {gt_label[i]}')

        return data_dict['img'], gt_label, ages


    def extract_feat(self, img):
        """Directly extract features from the backbone + neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, img_metas, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        # print_tensor('rawgt', gt_semantic_seg) # {key: [meta1, meta],}
        img, gt_label, age_step = self.update_img_metas(img, img_metas)
        x = self.extract_feat(img)
        losses = dict()
        loss, gap_feat1d = self.head.forward_train(x, gt_label, age_step, self.train_cfg)
        losses.update(loss)

        return losses

    def simple_test(self, img, img_metas, **kwargs):
        """Test without augmentation."""
        age_step = torch.tensor([m['img_meta_dict']['age'] for m in img_metas], 
                            dtype = img.dtype, device = img.device)
        x = self.extract_feat(img)
        out_cls, gap_feat1d = self.head.simple_test(x, age_step)

        if self.use_sigmoid:
            out_score = out_cls.float().sigmoid().cpu().numpy()
        else:
            out_score = out_cls.float().softmax(dim=1).cpu().numpy()
        # print_tensor('[SimpleTest]', out_score)
        return out_score


    def aug_test(self, imgs, img_metas, **kwargs):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        # to save memory, we get augmented seg logit inplace
        # print(img_metas)
        out_score = self.simple_test(imgs[0], img_metas[0], **kwargs)
        imgs[0] = None; torch.cuda.empty_cache()
        infer_times = 1
        # print('aug, post inference', seg_logit.shape)
        for i in range(1, len(imgs)):
            out_score_cur = self.simple_test(imgs[i], img_metas[i], **kwargs)
            # Cumulvate Moving Average: CMA_n+1 = (X_n+1 + n * CMA_n)/ (n + 1); CMA_n = (x1 + x2 + ...) / n
            out_score = (out_score_cur + out_score * infer_times) / (infer_times + 1)
            infer_times += 1
            imgs[i] = None; torch.cuda.empty_cache()
        return out_score


    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
        """
        if isinstance(imgs, torch.Tensor):
            imgs = [imgs]
        for var, name in [(imgs, 'imgs')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                            f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-3:])

        if len(imgs) == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)


    def forward_train_cl(self, img, img_metas, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.gpu_pipelines is not None:
            with torch.no_grad(): 
                image_list = []
                for _ in range(2):
                    mini_batch_holder = {'img': img.clone().detach()}
                    mini_batch_holder[f'img_meta_dict'] = [a[f'img_meta_dict'] for a in img_metas]
                    data_dict = self.aug_gpu_batch(mini_batch_holder)
                    image_list.append(data_dict.pop('img'))
                img1, img2 = image_list

        # print_tensor('image1', img1)
        # print_tensor('image2', img2)
        feat1 = self.extract_feat(img1)
        feat2 = self.extract_feat(img2)

        losses = dict()
        loss = self.head.forward_train(feat1, feat2)
        losses.update(loss)

        return losses
    
