
import torch
from torch import nn
from mmdet.core.post_processing.bbox_nms import batched_nms_3d
from mmdet.core.bbox.clip_nn import clip_boxes_to_image
from mmdet.core.bbox.ops_nn import remove_small_boxes
from monai.utils import fall_back_tuple
from monai.data.utils import compute_importance_map
from ...utils.resize import resize_3d, print_tensor
import copy, pdb


class BboxSegEnsembler1Case(object):

    def __init__(self, image_size, patch_size, num_classes = 2, test_cfg = {}, 
                 device = 'cpu', has_seg = True) -> None:
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.device = device
        self.test_cfg = test_cfg
        if has_seg:
            self.reset_seg_output()
            self.reset_seg_countmap()
        self.reset_det_storage()

    def initial_tile_weight_map(self, *args, **kwargs):
        self.tile_weight_map_3d = compute_importance_map(*args, 
                                                        device = self.device, 
                                                        **kwargs) #.clip(1e-8) #.to(torch.half)
        # print_tensor('[BuildTile] tile weight map', self.tile_weight_map_3d)
        # pdb.set_trace()

    def reset_det_storage(self):
        self.det_storage = []

    def reset_seg_output(self):
        if hasattr(self, 'seg_output_4d'):
            self.seg_output_4d = 0
        else:
            self.seg_output_4d = torch.zeros((self.num_classes, ) + self.image_size, 
                                            dtype=torch.float, device = self.device)

    def reset_seg_countmap(self):
        if hasattr(self, 'seg_countmap_4d'):
            self.seg_countmap_4d = 0
        else:
            self.seg_countmap_4d = torch.zeros((self.num_classes, ) + self.image_size, 
                                        dtype=torch.float, device = self.device)

    def _update_seg_output_1tile(self, seg_logit_tile, slice_tile, interpolate_mode = 'trilinear'):
        """[summary]

        Args:
            seg_logit_tile ([4d_tensor]): chwd, prediction of 1 tile/window 
            slice_tile ([type]): [description]
        """
        # pdb.set_trace()
        if self.tile_weight_map_3d.dtype != seg_logit_tile.dtype:
            self.tile_weight_map_3d = self.tile_weight_map_3d.to(seg_logit_tile.dtype)
        
        if seg_logit_tile.shape[-3:] != self.tile_weight_map_3d.shape:
            # print_tensor('[TileSeg] input ', seg_logit_tile)
            seg_logit_tile = resize_3d(seg_logit_tile[None], self.patch_size, 
                                        mode=interpolate_mode, 
                                        align_corners=False, 
                                        warning=False)[0]
            # print_tensor('[TileSeg] resize ', seg_logit_tile)

        self.seg_output_4d[slice_tile] = seg_logit_tile * self.tile_weight_map_3d[None, ...]
        self.seg_countmap_4d[slice_tile] += self.tile_weight_map_3d[None,  ...]
        # print_tensor(f'[BuildTile] segout', self.seg_output_4d)
        # print_tensor(f'[BuildTile] seg count', self.seg_countmap_4d)
        
    def update_seg_output_batch(self, seg_logit_tiles, slice_tiles, interpolate_mode = 'trilinear'):
        """
        Args:
            seg_logit_tiles (5d_tensor): bchwd, b for number of windows
            slice_tiles ([type]): slice for the b windows in the original image space
        preds_final = preds.new_full((bs, cs, resh, resw, resd), -16)
        preds_final[:, 0] = 16  # default [logit] 1hot vector for all pixels are (16, -16, ...)
        """
        # print_tensor(f'[BuildTile] seg logit tiles', seg_logit_tiles)
        if torch.isnan(seg_logit_tiles).any(): 
            print_tensor(f'[BuildTile] these tiles contains nan, so set to zero', seg_logit_tiles)
            seg_logit_tiles = seg_logit_tiles.new_full(seg_logit_tiles.shape, -16)
            if seg_logit_tiles.shape[1] > 1:  #for multi channels, 0th channel should set to 16 as logit
                seg_logit_tiles[:, 0] = 16
                
        for seg, sli in zip(seg_logit_tiles, slice_tiles):
            # print_tensor(f'[SegFill] seg tile slice {sli}', seg)
            # print_tensor(f'[SegFill] weight map', self.tile_weight_map_3d)
            self._update_seg_output_1tile(seg, sli[-4:], interpolate_mode)
    


    def finalize_segmap(self, final_slicing = None):
        # final_slicing : list[slice() ] len = 5
        result = (self.seg_output_4d/self.seg_countmap_4d)[None, ...]
        if final_slicing is not None:
            return result[final_slicing]
        else: return result

    def store_det_output(self, det_result_tiles):
        """
        args:
            det_result_tiles: list[(Tensor_nx7, Tensor_nx1)]
        """
        # det_result_tiles: list[(nx7, n), (), ...]
        self.det_storage.extend(det_result_tiles)
    
    def finalize_det_bbox(self, final_slicing_spatial = None, verbose = False):
        """
        # final_slicing_spatial : list[slice() ] len = 3
        return:
            dets_clean: tuple[Tensor(bboxes, nx7), Tensor(labels n)], 
        """
        bboxes_raw = torch.cat([a[0] for a in self.det_storage], axis = 0)
        labels_raw = torch.cat([a[1] for a in self.det_storage], axis = 0)

        if verbose and bboxes_raw.shape[0]> 0: 
            print_tensor('\n[SlideInfer] integrate det bbox raw', bboxes_raw[..., :6])
            print_tensor('\n[SlideInfer] integrate det score raw', bboxes_raw[..., 6])
            print_tensor('[SlideInfer] integrate det label raw', labels_raw)
        
        bboxes, labels = postprocess_detect_results((bboxes_raw, labels_raw), 
                                                        self.image_size, 
                                                        self.test_cfg)
                                                    
        if final_slicing_spatial is not None:
            for dim, slice_dim in enumerate(final_slicing_spatial):
                bboxes[:, [dim, dim + 3]] += - slice_dim[0]
                bboxes[:, [dim, dim + 3]] = torch.max(bboxes[:, [dim, dim + 3]], slice_dim[1])
        if verbose and bboxes.shape[0]> 0:  
            print_tensor('[SlideInfer] integrate det bbox refine', bboxes[..., :6])
            print_tensor('[SlideInfer] integrate det score refine', bboxes[..., 6])
            print_tensor('[SlideInfer] integrate det label refine', labels)
        return (bboxes, labels)


    def offset_preprocess4detect(self, det_result, img_meta, ready_img_shape, rescale):
        """
        store inplace
        preprocess pipeline: resize, padding, flip
        """

        this_bboxes_raw, this_label = det_result # (bboxes_nx7, labels_nx1)
        this_bboxes = this_bboxes_raw.clone().detach()
        img1info = img_meta['img_meta_dict']
        origin_shape = img1info['spatial_shape']
        shape_post_resize = img1info.get('shape_post_resize', origin_shape)
        shape_post_pad = img1info.get('padshape', shape_post_resize) # pad to end
        # raw_right = sanity_check_bbox(this_bboxes_raw)
        # print(f'\n[PostDet] ori {origin_shape} resize{shape_post_resize} pad{shape_post_pad} boxright {raw_right}')

        # 1. accounting for flip
        flip = img1info.get('flip', False)
        flip_direction = img1info.get('flip_direction', None)
        if flip_direction == 'diagonal': flip_dims = (0, 1)
        elif flip_direction == 'headfeet': flip_dims = (2, )
        elif flip_direction == 'all3axis': flip_dims = (0, 1, 2)
        else: flip_dims = None

        if flip and (flip_dims is not None): 
            for fdim in flip_dims:
                # when flipping the bbox, large become small and small large, 
                # then their position should also change accordingly, small to front and large to the end
                this_bboxes[:, [fdim, fdim + 3]] = ready_img_shape[fdim] - this_bboxes_raw[:, [fdim + 3, fdim]]

        # flip_right = sanity_check_bbox(this_bboxes)
        # if not flip_right:
        #     pdb.set_trace()
        #     flip_roi_mask = roi_sanity_mask(this_bboxes)
        #     raw_roi_mask = roi_sanity_mask(this_bboxes_raw)
        # 2, accounting for padding     
        this_bboxes[:, :6] = clip_boxes_to_image(this_bboxes[:, :6], shape_post_pad, is_xyz=True)

        # 3. accounting for resizing
        if img1info.get('new_pixdim', False):
            if rescale and (ready_img_shape != origin_shape):
                for i, (f, o) in  enumerate(zip(shape_post_resize, origin_shape)):
                    this_bboxes[:, [i, i+3]] = this_bboxes[:, [i, i+3]] * o/f
        
        # print_tensor('[SlideIn] postprocess bbox', this_bboxes[:, :6])
        # print_tensor('[SlideIn] postprocess score', this_bboxes[:, 6])
        # print_tensor('[SlideIn] postprocess label', this_label)
        return (this_bboxes, this_label)

    def offset_preprocess4seg(self, seg_result, img_meta, ready_img_shape, rescale):
        """
        store inplace
        preprocess pipeline: origin_shape > resize, padding, flip > ready_img_shape
        """
        this_seg_5d = seg_result

        img1info = img_meta['img_meta_dict']
        origin_shape = img1info['spatial_shape']
        shape_post_resize = img1info.get('shape_post_resize', origin_shape)
        shape_post_pad = img1info.get('padshape', shape_post_resize) # pad to end

        # 1. accounting for flip
        flip = img1info.get('flip', False)
        flip_direction = img1info.get('flip_direction', None)
        if flip_direction == 'diagonal': flip_dims = (0, 1)
        elif flip_direction == 'headfeet': flip_dims = (2, )
        elif flip_direction == 'all3axis': flip_dims = (0, 1, 2)
        else: flip_dims = None

        if flip and (flip_dims is not None): 
            this_seg_5d = this_seg_5d.flip(dims = [a + 2 for a in flip_dims])

        # 2, accounting for padding
        # print_tensor(f'\t[SlideInfer] ShapePostResize:{shape_post_resize} OriginShape: {origin_shape} Preds:', preds)       
        this_seg_5d = this_seg_5d[...,:shape_post_pad[-3], 
                                        :shape_post_pad[-2], 
                                        :shape_post_pad[-1]]

        # 3. accounting for resizing
        if img1info.get('new_pixdim', False):
            if rescale and (shape_post_resize != origin_shape):
                this_seg_5d = resize_3d(this_seg_5d, size=origin_shape,
                                        mode='trilinear', 
                                        align_corners=False,
                                        warning=False)
        # print_tensor('[SlideIn] postprocess seg', this_seg_5d)
        return this_seg_5d

roi_sanity_mask = lambda x : ((x[:, [3, 4, 5]]  - x[:, [0, 1, 2]]) < 1).sum(dim = 1).bool()

def sanity_check_bbox(bbox_nx7):
    assert len(bbox_nx7.shape) == 2, f'bbox dim should be 2 but got {bbox_nx7.shape}'
    if bbox_nx7.shape[0] < 1: return True
    sanity_nx1 = roi_sanity_mask(bbox_nx7)
    # pdb.set_trace()
    return not sanity_nx1.any().item()


def postprocess_detect_results(det_tuple, img_shape, test_cfg, verbose = False):
    """
    Args:
        det_tuples: list[tuple[nx7, n]], bboxes, labels
        cfg.score_thr, cfg.nms, cfg.max_per_img
    
    """
    test_cfg = copy.copy(test_cfg)
    max_per_img = test_cfg.pop('max_per_img', 100)
    # 1. top 
    # from mmdet.core.export import get_k_for_topk
    bbox_nx7, label_n = det_tuple
    if bbox_nx7.shape[0]== 0:
        return det_tuple

    if verbose: 
        print_tensor('[WholeImage] all bbox', bbox_nx7[:, :6])
        print_tensor('[WholeImage] all scores', bbox_nx7[:, 6])

    topk = test_cfg.pop('nms_pre_tiles', 200)
    topk = min(topk, bbox_nx7.shape[0])

    _, topk_idx = bbox_nx7[:, -1].topk(topk)
    bbox_nx7, label_n = bbox_nx7[topk_idx, ...], label_n[topk_idx]

    # pdb.set_trace()
    # 2. score threshold
    # NonZero not supported  in TensorRT
    valid_mask = bbox_nx7[:, 6] >  test_cfg.pop('score_thr', 0.01) # score_thr
    bbox_nx7, label_n = bbox_nx7[valid_mask], label_n[valid_mask]
    if bbox_nx7.shape[0] == 0: return bbox_nx7, label_n

    # 3. remove small boxes
    keep = remove_small_boxes(bbox_nx7[:, :6], min_size=test_cfg.pop('min_bbox_size', 2), is_xyz=True)
    bbox_nx7, label_n = bbox_nx7[keep], label_n[keep]
    if bbox_nx7.shape[0] == 0: return bbox_nx7, label_n

    # 4. clipping
    bbox_nx7[:, :6] = clip_boxes_to_image(bbox_nx7[:, :6], img_shape, is_xyz = True)

    # 5. nms
    dets_clean, keep_idx = batched_nms_3d(bbox_nx7[:, :6], bbox_nx7[:, 6], 
                                         label_n, test_cfg['nms'])

    label_n = label_n[keep_idx]

    if max_per_img > 0:
        dets_clean = dets_clean[:max_per_img]
        label_n = label_n[:max_per_img]
    
    if verbose: 
        print_tensor('[WholeImage] remain bbox', dets_clean[:, :6])
        print_tensor('[WholeImage] remain score', dets_clean[:, 6])
    return dets_clean, label_n

class ShapeHolder:

    def __init__(self, ready_img_shape, patch_shape) -> None:
        self.ready_img_shape = ready_img_shape # (HWD)
        self.spatial_dim = len(ready_img_shape)

        self.patch_shape = fall_back_tuple(patch_shape, ready_img_shape)

        # # in case that image size is smaller than roi size
        self.ready_shape_safe = tuple(max(self.ready_img_shape[i], self.patch_shape[i]) 
                                            for i in range(self.spatial_dim))

        # torch padding from last dim to first dim, backward
        self.pad_shape = []
        for k in reversed(range(self.spatial_dim)):
            diff = max(self.patch_shape[k] - self.ready_img_shape[k], 0)
            half = diff // 2
            self.pad_shape.extend([half, diff - half])
        
        self._get_final_slicing()

    def _get_final_slicing(self):
        
        self.final_slicing = []
        for sp in range(self.spatial_dim): # 0 2 4 
            slice_dim = slice(self.pad_shape[sp * 2], 
                              self.ready_img_shape[self.spatial_dim - sp - 1] + self.pad_shape[sp * 2])
            self.final_slicing.insert(0, slice_dim)
        self.final_slicing_spatial = self.final_slicing
        while len(self.final_slicing) < len(self.ready_shape_safe):
            self.final_slicing.insert(0, slice(None))
        


def _get_scan_interval(
    image_size, roi_size, num_spatial_dims: int, overlap: float):
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)
