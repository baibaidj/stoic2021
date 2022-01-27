
import torch
import torch.nn.functional as F

from mmdet.utils.resize import resize_3d
from ..builder import DETECTORS, build_backbone, build_neck, build_head
from .base_learner_3d import BaseLearner3D
from ...datasets.pipelines import Compose
from mmdet.models.necks.fpn_3d import FPN3D
from mmdet.models.semantic_heads import FCNHead3D
from mmdet.models.backbones.swin_3d import SwinTransformer3D4SimMIM
from monai.data.utils import dense_patch_slices
from mmdet.core.evaluation.slide_window_infer import (
            _get_scan_interval, ShapeHolder, BboxSegEnsembler1Case)
from mmdet.utils import print_tensor
# from torch import distributed as dist
import copy, ipdb

@DETECTORS.register_module()
class SimMIM(BaseLearner3D):
    def __init__(self, 
                 backbone,
                 neck=None,
                 seg_head = None,
                 gpu_aug_pipelines = [],
                 train_cfg=None,
                 test_cfg=None,
                 mask_cfg = dict(input_size=(160, 160, 160), mask_patch_size=32,
                                 stem_stride=4, mask_ratio=0.6), 
                pretrained = None, 
                init_cfg = None, 
                verbose = False):
        super(SimMIM, self).__init__(init_cfg)

        self.backbone : SwinTransformer3D4SimMIM = build_backbone(backbone)
        if neck is not None:
            self.neck : FPN3D = build_neck(neck)

        self.seg_head: FCNHead3D = build_head(seg_head) if seg_head is not None else None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.verbose = verbose
        self.in_chans = self.backbone.in_channels
        self.stem_stride = self.backbone.strides[0]
        self.mask_generator = MaskGeneratorND(**mask_cfg)

        self.gpu_pipelines = Compose(gpu_aug_pipelines)

        self.roi_size = self.test_cfg.pop('roi_size', None)
        self.sw_batch_size = self.test_cfg.pop('sw_batch_size', 6)
        self.overlap = self.test_cfg.pop('overlap', 0.05)
        self.sigma_scale = self.test_cfg.pop('sigma_scale', 0.125)

    def create_batch_mask(self, num_sample, device):
        mask_list = []
        for i in range(num_sample):
            mask_i = self.mask_generator(device)
            mask_list.append(mask_i)
        masks = torch.stack(mask_list)
        return masks

    @torch.no_grad()
    def update_img_metas(self, imgs, img_metas, **kwargs):
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        mini_batch_holder = {'img': imgs}
        mini_batch_holder[f'img_meta_dict'] = [a[f'img_meta_dict'] for a in img_metas]
        data_dict = self.gpu_pipelines(mini_batch_holder)
        imgs = data_dict['img']
        masks = self.create_batch_mask(len(img_metas), imgs.device)
        return imgs, masks
    
    def forward_train(self, img, img_metas, **kwargs):
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())

        x, mask = self.update_img_metas(img, img_metas)

        # if self.verbose: 
        #     print_tensor('\nIMG raw', x)
        #     print_tensor('IMG mask', mask)

        feat = self.extract_feat(x, mask)
        x_rec = self.seg_head.forward_test(feat, img_metas, self.test_cfg)

        if self.verbose: 
            print_tensor(f'\n[SimMim] x input', x)
            print_tensor(f'[SimMim] reconstruct', x_rec)
            print_tensor(f'[SimMim] mask reshape', mask)
        
        if x_rec.shape != x.shape: 
            x_rec = resize_3d(x_rec, size = x.shape[2:], mode = 'trilinear', align_corners=False)
        # ops = torch.repeat_interleave()
        mask = mask.repeat_interleave(self.stem_stride, 1
                    ).repeat_interleave(self.stem_stride, 2
                    ).repeat_interleave(self.stem_stride, 3).unsqueeze(1).contiguous()

        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        losses = {'loss_rec' : loss}
        return losses

    def extract_feat(self, img, mask = None):
        """Directly extract features from the backbone + neck
        """
        x = self.backbone(img, mask = mask)
        if self.with_neck:
            x = self.neck(x)
        return x

    def simple_test_tile(self, img, img_metas, mask = None):
        """Test without augmentation."""
        z = self.extract_feat(img, mask)
        x_rec = self.seg_head.forward_test(z, img_metas, self.test_cfg)
        
        if x_rec.shape[2:] != img.shape[2:]:
            x_rec = resize_3d(x_rec, size = img.shape[2:], mode = 'trilinear', align_corners=False)

        return x_rec

    def slice_window_infer(self, inputs: torch.Tensor,
                            img_metas, 
                            mode = 'constant',
                            sigma_scale = 0.125,
                            padding_mode = 'constant',
                            rescale = True):
        # ip_dtype = torch.float #inputs.dtype
        device = inputs.device; sw_device = device#'cpu'
        batch_size = inputs.shape[0]
        Shaper = ShapeHolder(inputs.shape[2:], self.roi_size)
        scan_interval = _get_scan_interval(Shaper.ready_shape_safe, Shaper.patch_shape, 
                                           Shaper.spatial_dim, self.overlap)
        window_slices = dense_patch_slices(Shaper.ready_shape_safe, Shaper.patch_shape, scan_interval)
        num_win = len(window_slices)  # number of windows per image
        # Perform predictions
        # print_tensor('original inputs', inputs) BCHWD
        rec_image_full = inputs.new_full(inputs.shape, fill_value=-5)
        rec_mask_full = inputs.new_zeros(inputs.shape)
        
        for bix in range(batch_size):
            img_meta = img_metas[bix]
            for six in range(0, num_win, self.sw_batch_size):
                slice_range = range(six, min(six + self.sw_batch_size, num_win))
                tiles_slicer = [[slice(bix, bix + 1), slice(None)] + list(window_slices[idx]) 
                                                                    for idx in slice_range]
                img_meta_tiles = [copy.deepcopy(img_meta)  for _ in range(len(slice_range))]#[img_meta] * len(slice_range)
                for i, m in enumerate(img_meta_tiles): 
                    m['tile_origin'] = [window_slices[slice_range[i]][d].start for d in range(Shaper.spatial_dim)]
                    m['img_shape'] = None #Shaper.patch_shape
                    m['scale_factor'] = [1 for _ in range(Shaper.spatial_dim * 2)]
                
                window_data = torch.cat([inputs[win_slice] for win_slice in tiles_slicer]).to(device)
                rand_masks_4d = self.create_batch_mask(len(img_meta_tiles), device)
                # print_tensor(f'[SlideInfer] window {six} data', window_data)
                rec_imgs = self.simple_test_tile(window_data, img_meta_tiles, mask = rand_masks_4d)  

                rand_masks_5d =  rand_masks_4d.repeat_interleave(self.stem_stride, 1
                                                ).repeat_interleave(self.stem_stride, 2
                                                ).repeat_interleave(self.stem_stride, 3)[:, None, ...].contiguous()
                # batched patch segmentation
                # Cumulvate Moving Average: CMA_n+1 = (X_n+1 + n * CMA_n)/ (n + 1); CMA_n = (x1 + x2 + ...) / n
                for j, slicer in enumerate(tiles_slicer):
                    rec_image_full[slicer] = rec_imgs[j:j+1]
                    rec_mask_full[slicer] = rand_masks_5d[j:j+1]
            torch.cuda.empty_cache()
            # print_tensor('[Ensemble] reset', ensembler.seg_output_4d)
            # print_tensor('[Ensemble] reset', ensembler.seg_countmap_4d)
        return rec_image_full, rec_mask_full

    def simple_test(self, img, img_metas, rescale = False):
        """Test without augmentation."""

        rec_image_full, rec_mask_full = self.slice_window_infer(img, img_metas)

        return rec_image_full, rec_mask_full


    def aug_test(self, imgs, img_metas, rescale=False):

        feats = self.extract_feats(imgs)
        outs = [self.seg_head.forward_test(a) for a in feats]
        return outs

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def product(vector):
    result = 0
    if isinstance(vector, (int, float)):
        return vector
    elif isinstance(vector, (list, tuple)):
        num_ = len(vector)
        if num_ == 0:
            return
        elif num_ == 1:
            return vector[1]
        else:
            result = vector[0]
            for n in vector[1:]:
                result = result * n
    else:
        raise ValueError(f'Input should be number but got {type(vector)}')
    return result
    
import numpy as np

class MaskGeneratorND:
    def __init__(self, input_size=(192, 192), 
        mask_patch_size=32, stem_stride=4, mask_ratio=0.6, device = 'cpu', 
        verbose = True):
        """
        ND: N-dimensional, N can assume 2 or 3
        Args:
            model_patch_size: the stride of stem layer in Swin
        """
        assert isinstance(input_size, (tuple, list)), \
            f'input size should be tuple or list but got {type(input_size)}'

        # self.add_key = add_key
        # self.img_key = img_key
        self.input_size = input_size
        self.dim = len(input_size)
        self.num_pixel = product(input_size)
        self.mask_patch_size = mask_patch_size
        self.stem_stride = stem_stride
        self.mask_ratio = mask_ratio
        self.device = device
        self.verbose = verbose

        assert self.num_pixel % self.mask_patch_size == 0
        assert self.mask_patch_size % self.stem_stride == 0
        
        self.token_map_size = tuple([s // self.mask_patch_size for i, s in enumerate(self.input_size)])
        self.scale = self.mask_patch_size // self.stem_stride
        
        self.token_count = product(self.token_map_size)
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

        if self.verbose:
            print(f'[MaskGen] input size {input_size} mask patch {self.mask_patch_size} mask ratio {mask_ratio}', 
                  f'\n token map {self.token_map_size} token count {self.token_count} mask count {self.mask_count}'  )

    def __call__(self, device = 'cuda:0'):
        device = device if device is not None else self.device
        mask = torch.zeros(self.token_count, dtype=torch.long, device = device)
        mask_idx = torch.randperm(self.token_count, device=device)[:self.mask_count]
        mask[mask_idx] = 1

        mask = mask.reshape(self.token_map_size) # 6x6?
        mask = torch.repeat_interleave(mask, self.scale, dim=0).repeat_interleave(self.scale, dim=1)
        if self.dim == 3: 
            mask = mask.repeat_interleave(self.scale, dim = 2) # 48x48?
        return mask
