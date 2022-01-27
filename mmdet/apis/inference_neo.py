import torch, copy

from mmdet.datasets.builder import PIPELINES
from mmcv.utils import build_from_cfg
from mmdet.core import bbox2result3d, batched_nms_3d
import ipdb

def tta_detect_1by1(model, img, affine = None, rescale = True,
                            need_probs = True, guide_mask = None, target_spacings = [None]):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        result_prob: post softmax/sigmoid before binarization
        feature_map: 
        data: 

    """

    cfg = copy.deepcopy(model.cfg)
    device = next(model.parameters()).device  # model device
    # 0. ToTensor ToGPU add channel
    data_dict = LoadImageGPU()(dict(img=img, affine = affine, seg = guide_mask), device = device)
    normalizer = build_from_cfg(dict(type='NormalizeIntensityGPUd',
                                    keys='img',
                                    subtrahend=330,
                                    divisor=562.5,
                                    percentile_99_5=3071,
                                    percentile_00_5=-927), PIPELINES)
    # pdb.set_trace()
    # 1. normalize image
    # collect_keymap = {'img' : 'img', 'seg': 'seg', 'img_metas': 'img_meta_dict'}
    data_dict = normalizer(data_dict)
    det_results_tta = []
    # NOTE: Try two spacing, rather than two view
    for new_spacing in target_spacings:
        for flip_direction in [None, ]: #, 'diagonal'
            # if new_spacing is None and flip_direction is not None:
            #     continue
            print(f'\n[DetTTA] new spacing {new_spacing}  flip {flip_direction}')
            resizer = ResizeTensor5DGPU(keys = ('img', 'seg'), new_spacing = new_spacing)
            flipper = FlipTensor5DGPU(keys = ('img', 'seg'), flip_direction = flip_direction)
            # 1. respacing
            data_var = resizer(**data_dict)
            # 2. FlipTTA
            data_var = flipper(**data_var)
            # outer most : tta ;  2nd outer sample/minibatch; inner meta for 1 image
            data_var.pop('seg_meta_dict', None)
            data_var['img_metas'] = [{'img_meta_dict': data_var.pop('img_meta_dict', None)}]
            for k in data_var.keys(): data_var[k] = [data_var[k]]

            # forward the model
            with torch.no_grad():
                # see models/segmentors/base.py 108, forward method
                det_results, *seg_results = model(return_loss=False, 
                                                rescale=rescale, 
                                                need_probs = need_probs, 
                                                **data_var)
            det_results_tta.extend(det_results)
            # del data_var; torch.cuda.empty_cache()
    # detection
    bbox_nx7 = torch.cat([d1[0] for d1 in det_results_tta], axis = 0) # nx7
    label_nx1 = torch.cat([d1[1] for d1 in det_results_tta], axis = 0) # nx1

    if bbox_nx7.shape[0] < 1: 
    # pdb.set_trace()
        return [[torch.zeros((0, 7)), ] ], None
        
    dets_bbox_nx7, keep_idx = batched_nms_3d(bbox_nx7[:, :6], bbox_nx7[:, 6], 
                                            label_nx1, cfg.model.test_cfg['nms'])
    label_nx1 = label_nx1[keep_idx]
    max_per_img, num_classes = cfg.model.test_cfg['max_per_img'], cfg.num_classes
    if max_per_img > 0:
        dets_bbox_nx7 = dets_bbox_nx7[: max_per_img]
        label_nx1 = label_nx1[:max_per_img]
    # pdb.set_trace()
    bbox_results = [bbox2result3d(dets_bbox_nx7, label_nx1, num_classes)]
    return bbox_results, seg_results

def masked_image_modeling(model, img, affine = None, rescale = True,
                          guide_mask = None, target_spacing = (1.6, 1.6, 1.6),
                          input_patch_size = (192, 192, 160),  
                          norm_intense_cfg = dict(type='NormalizeIntensityGPUd',
                                    keys='img',
                                    subtrahend=330,
                                    divisor=562.5,
                                    percentile_99_5=3071,
                                    percentile_00_5=-927)):

    device = next(model.parameters()).device  # model device
    # 0. ToTensor ToGPU add channel
    data_dict = LoadImageGPU()(dict(img=img, affine = affine, seg = guide_mask), device = device)

    normalizer = build_from_cfg(norm_intense_cfg, PIPELINES)
    resizer = ResizeTensor5DGPU(keys = ('img',), new_spacing = target_spacing)
    cropper = SpatialCrop5DGPU(keys = ('img',), roi_size = input_patch_size)

    # pdb.set_trace()
    # 1. normalize image
    data_dict = normalizer(data_dict)
    data_var = resizer(**data_dict)
    data_var = cropper(**data_var)
    # outer most : tta ;  2nd outer sample/minibatch; inner meta for 1 image
    data_var.pop('seg_meta_dict', None)
    data_var.pop('seg', None)
    data_var['img_metas'] = [{'img_meta_dict': data_var.pop('img_meta_dict', None)}]
    for k in data_var.keys(): data_var[k] = [data_var[k]] 

    # forward the model
    with torch.no_grad():
        # see models/segmentors/base.py 108, forward method
        # ipdb.set_trace()
        reconstr_images, rand_masks = model(return_loss=False, 
                                        rescale=rescale, 
                                        **data_var)

    img_meta_dict = data_var['img_metas'][0][0]['img_meta_dict'] # aug_index, batch_index
    slices4img = img_meta_dict['slices4img']
    slices4patch =  img_meta_dict['slices4patch']
    shape_post_resize = img_meta_dict['shape_post_resize']
    shape_origin = img_meta_dict['spatial_shape']
    
    # 1. anti_crop
    reconstr_image_result = reconstr_images.new_full([1, 1] + shape_post_resize, 
                                                    reconstr_images.min(), 
                                                    dtype = reconstr_images.dtype)
    rand_mask_result = rand_masks.new_full([1, 1] + shape_post_resize, 
                                            rand_masks.min(), 
                                            dtype = rand_masks.dtype)

    reconstr_image_result[slices4img] = reconstr_images[slices4patch]
    rand_mask_result[slices4img] = rand_masks[slices4patch]
    # 2. anti_resize
    reconstr_image_result = F.interpolate(reconstr_image_result, size = shape_origin, 
                                            mode = 'trilinear', align_corners=False)

    rand_mask_result = F.interpolate(rand_mask_result, size = shape_origin, 
                                     mode = 'nearest', align_corners=None)      
    return reconstr_image_result, rand_mask_result


class LoadImageGPU:
    """A simple pipeline to load image."""

    def __call__(self, results, device = 'cuda:0'):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        results['img_meta_dict'] = dict(original_affine = results['affine'], 
                                        affine = results.pop('affine', None),
                                        spatial_shape = results['img'].shape ,
                                        filename_or_obj = results.get('filename', ''), 
                                        # new_pixdim = None, 
                                        # flip = False, 
                                        # flip_direction = 'diagnal'
                                        )
        results['seg_meta_dict'] = dict(**results['img_meta_dict'])

        results['img'] = torch.from_numpy(results['img']).float().to(device)[None, None]
        if results['seg'] is not None:
            results['seg'] = torch.from_numpy(results['seg']).float().to(device)[None, None]
        return results


import torch.nn.functional as F
class ResizeTensor5DGPU(object):
    """
    Resize 5D tensor on GPU
    """

    def __init__(
        self,
        keys,
        new_spacing ,
        mode = ['trilinear', 'nearest'],
        align_corners = [False, False],
        dtype = [torch.float, torch.long],
        meta_key_postfix: str = "meta_dict",
        verbose = False,

    ) -> None:
        """
        Args:

        Raises:
            TypeError: When ``meta_key_postfix`` is not a ``str``.

        """
        super().__init__()
        self.keys = keys
        self.new_spacing = new_spacing
        self.mode = mode
        self.align_corners = align_corners #, len(self.keys))
        # self.dtype = dtype # ensure_tuple_rep(dtype, len(self.keys))
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_key_postfix = meta_key_postfix
        self.verbose = verbose

    def __call__(self, **data):
        d = dict(data)
        # new_pixdim = d['img_meta_dict'].get('new_pixdim', None)
        # print('Check new pixdim', new_pixdim)
        if self.new_spacing is None: return d    

        for idx, key in enumerate(self.keys):
            meta_data = d[f"{key}_{self.meta_key_postfix}"]
            # resample array of each corresponding key
            # using affine fetched from d[affine_key]
            old_shape = d[key].shape[2:]
            old_spacing = [abs(meta_data['affine'][i, i]) for i in range(3)]
            new_shape = [round(s * old_spacing[i] / self.new_spacing[i]) for i, s in enumerate(old_shape)]

            d[key] = F.interpolate(d[key], size = new_shape, 
                                    mode=self.mode[idx], 
                                    align_corners=self.align_corners[idx])
            if self.verbose: print(f'\tRespacing: from {old_shape} {old_spacing} to {new_shape} {self.new_spacing}')
            # set the 'affine' key
            # meta_data["affine"] = self.new_spacing #TODO: get it right
            meta_data['shape_post_resize'] = new_shape
        return d


class FlipTensor5DGPU(object):
    """
    
    Flip Last three dimensions which are the spatial ones
    """
    def __init__(
        self,
        keys,
        flip_direction = 'diagnoal',
        meta_suffix = 'meta_dict'
    ) -> None:

        self.keys = keys
        # self.flip = flip
        self.flip_direction = flip_direction
        self.meta_suffix = meta_suffix

    def __call__( self, **data):
        # self.randomize()
        d = dict(data)
        # flip = d['img_meta_dict']['flip']
        # flip_direction = d['img_meta_dict']['flip_direction']

        if self.flip_direction == 'diagonal': flip_dims = (2, 3)
        elif self.flip_direction == 'headfeet': flip_dims = (4, )
        elif self.flip_direction == 'all3axis': flip_dims = (2, 3, 4)
        else: flip_dims = None
        # print(f'PIPELINE: Flip {flip} Direction {flip_direction} dims {flip_dims}')
        if flip_dims is not None: 
            for key in self.keys:
                d[key] = torch.flip(d[key], flip_dims)
                d[f'{key}_{self.meta_suffix}']['flip'] = None
                d[f'{key}_{self.meta_suffix}']['flip_direction'] = self.flip_direction
                
        return d



class SpatialCrop5DGPU(object):
    """
    General purpose cropper to produce sub-volume region of interest (ROI).
    It can support to crop ND spatial (channel-first) data.
    Either a spatial center and size must be provided, or alternatively,
    if center and size are not provided, the start and end coordinates of the ROI must be provided.
    """

    def __init__(
        self,
        keys, 
        roi_size, 
        meta_suffix = 'meta_dict', 
    ) -> None:
        """
        Args:
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI.
            roi_start: voxel coordinates for start of the crop ROI.
            roi_end: voxel coordinates for end of the crop ROI.
        """
        self.keys = keys
        self.meta_suffix = meta_suffix
        self.roi_size = torch.tensor(roi_size, dtype=torch.long)
        # self.initialize_param(roi_center)

    def initialize_param(self, roi_center):
        roi_center = torch.tensor(roi_center, dtype=torch.long)
        self.roi_start = roi_center - torch.floor_divide(self.roi_size, 2)
        self.roi_end = self.roi_start + self.roi_size

    def __call__(self, **data):
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        # print('img', img.shape)
        d = dict(data)
        b, c, *img_size_3d = d[self.keys[0]].shape
        
        image_shape = torch.tensor(img_size_3d, dtype = torch.long)

        self.initialize_param([s //2 for s in image_shape])

        end_xyz_delta = self.roi_end - image_shape
        patch_shape = tuple(self.roi_size)
        start_xyz_img = self.roi_start.clip(0) 
        end_xyz_img = self.roi_end.minimum(image_shape)
        # slice4img = [slice(start_xyz_img[i], end_xyz_img[i]) for i in range(3)]

        start_xyz_patch = torch.where(self.roi_start < 0, torch.abs(self.roi_start), 0)
        end_xyz_patch = torch.where(end_xyz_delta > 0, self.roi_size - torch.abs(end_xyz_delta), self.roi_size) #
        # slice4patch = [slice(start_xyz_patch[i], end_xyz_patch[i]) for i in range(3)]

        sd = min(len(self.roi_start), len(self.roi_end), len(image_shape))  # spatial dims
        slices4img = tuple( [slice(None), slice(None)] + 
                            [slice(s, e) for s, e in zip(start_xyz_img[:sd], end_xyz_img[:sd])])
        slices4patch = tuple([slice(None), slice(None)] + 
                             [slice(s, e) for s, e in zip(start_xyz_patch[:sd], end_xyz_patch[:sd])])

        for k in self.keys:
            img_k : torch.Tensor = d[k]
            patch_image = img_k.new_full(img_k.shape[:2] + patch_shape, img_k.min(), dtype = img_k.dtype)
            patch_image[slices4patch] = img_k[slices4img]
            d[k] = patch_image
            d[f'{k}_{self.meta_suffix}']['slices4img'] = slices4img
            d[f'{k}_{self.meta_suffix}']['slices4patch'] = slices4patch

        return d