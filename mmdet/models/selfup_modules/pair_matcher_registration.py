import torch.nn as nn
import torch.nn.functional as F
import torch

from ..builder import DETECTORS, build_backbone, build_neck, build_head
from ..detectors.base3d import BaseDetector3D
from ...datasets.pipelines import Compose
from ..semantic_heads.match_head import FlowHead3D
from mmdet.utils.resize import list_dict2dict_list, resize_3d, print_tensor

import pdb

# from mmcv.runner import auto_fp16

get_meta_dict  = lambda img_meta: img_meta[0]['img_meta_dict'] \
                if isinstance(img_meta, (list, tuple)) else img_meta['img_meta_dict']

@DETECTORS.register_module()
class PairMatcher3D(BaseDetector3D):
    """Encoder Decoder segmentors for medical imaging.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    use_sdm_aux: [bool] if building an auxillary head that predicts signed distance map as extra supervision
    use_tsm: [bool] if using temporal shift module to 
    
    """
    def __init__(self, backbone,
                    decode_head,
                    neck=None,
                    auxiliary_head=None,
                    train_cfg=None,
                    test_cfg=None,
                    gpu_aug_pipelines = [], 
                    init_cfg = None, 
                    **kwargs):
        super(PairMatcher3D, self).__init__(init_cfg=init_cfg)

        self.backbone = build_backbone(backbone)
        if neck is not None: 
            self.neck = build_neck(neck)
        
        self.decode_head : FlowHead3D = build_head(decode_head)

        self.gpu_pipelines = Compose(gpu_aug_pipelines) if gpu_aug_pipelines is not None else None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        if self.with_seghead: mask = self.seg_head(x)
        else: mask = None
        return outs, mask

    @torch.no_grad()
    def update_img_metas(self, imgs, img_metas, seg, **kwargs):
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        # TODO: adjust keys
        gt_keys = ['img', 'seg'] # 'img_metas'
        data_dict = list_dict2dict_list(img_metas, verbose=False)
        data_dict.update({'img': imgs, 'seg': seg})
        data_dict = self.gpu_pipelines(data_dict)
        # for b, m in enumerate(img_metas): m['patch_shape'] = data_dict['patch_shape']
        return [data_dict[k] for k in gt_keys] 

    def forward_train(self,
                      img,
                      img_metas,
                      seg,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W, D).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, tl_z, br_x, br_y, br_z] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        img, gt_bboxes, gt_labels, gt_semantic_seg = self.update_img_metas(
                                            img, img_metas, seg)
        x = self.extract_feat(img)
        losses = self.decode_head.forward_train(x, img_metas, img, self.train_cfg)

        # pdb.set_trace()
        return losses

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        preds, *aux_output = self.encode_decode(img, img_meta)
        img_meta_dict = get_meta_dict(img_meta)
        # print(img_meta[0])
        target_shape = img_meta_dict['spatial_shape'] # TODO: determine the key word in image_meta_dict for original shape
        if img_meta_dict.get('new_pixdim', False):
            if rescale and (preds.shape[-3:] != target_shape):    
                preds = resize_3d(
                    preds,
                    size=target_shape,
                    mode=self.get_output_mode(), #'bilinear'
                    align_corners=self.align_corners,
                    warning=False)
        return preds


    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        
        x = self.extract_feat(img)

        flow_field_final, moved_image = self.decode_head.simple_test(x, img_meta, img, rescale=True)

        return moved_image, flow_field_final

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got '
                                f'{type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) != '
                             f'num of image meta ({len(img_metas)})')
        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        # for img_meta in img_metas:
        #     ori_shapes = [_['ori_shape'] for _ in img_meta]
        #     assert all(shape == ori_shapes[0] for shape in ori_shapes)
        #     img_shapes = [_['img_shape'] for _ in img_meta]
        #     assert all(shape == img_shapes[0] for shape in img_shapes)
        #     pad_shapes = [_['pad_shape'] for _ in img_meta]
        #     assert all(shape == pad_shapes[0] for shape in pad_shapes)
        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)


    # def aug_test(self, imgs, img_metas, rescale=True, need_logits = False, aux_head_index = None):
    #     """Test with augmentations.

    #     Only rescale=True is supported.
    #     """
    #     # aug_test rescale all imgs back to ori_shape for now
    #     assert rescale
    #     # to save memory, we get augmented seg logit inplace
    #     # print(img_metas)
    #     seg_logit, aux_output = self.inference(imgs[0], img_metas[0], rescale, aux_head_index)
    #     # print('aug, post inference', seg_logit.shape)
    #     for i in range(1, len(imgs)):
    #         cur_seg_logit, cur_aux_output = self.inference(imgs[i], img_metas[i], rescale, aux_head_index)
    #         if aux_head_index is not None: aux_output += cur_aux_output
    #         seg_logit += cur_seg_logit
    #     seg_logit /= len(imgs)
    #     if aux_head_index is not None: aux_output /= len(imgs)
    #     if need_logits:
    #         seg_pred = seg_logit.cpu() #.float().cpu().numpy()
    #     else:
    #         seg_pred = seg_logit.argmax(dim=1) if seg_logit.shape[1] > 1 else seg_logit.squeeze(1) > 0.5
    #         seg_pred = seg_pred.int().cpu().numpy()
    #         # print('aug, post argmax', seg_pred.shape)        
    #         # unravel batch dim
    #         seg_pred = list(seg_pred)
    #     if aux_head_index is not None:  return seg_pred, aux_output
    #     else: return seg_pred