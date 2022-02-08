

from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch


import os, random
import pandas as pd

from monai.config import KeysCollection
from monai.transforms.croppad.array import SpatialCrop, SpatialPad, Method, NumpyPadMode
from monai.transforms.transform import Randomizable, Transform, MapTransform

from monai.utils import ensure_tuple_rep, fall_back_tuple

randint = lambda x: random.randrange(-x, x)
print_tensor = lambda n, x: print(n, type(x), x.dtype, x.shape, x.min(), x.max())


NumpyPadModeSequence = Union[Sequence[Union[NumpyPadMode, str]], NumpyPadMode, str]
class SpatialPadd_(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SpatialPad`.
    Performs padding to the data, symmetric for all sides or all on one side for each dimension.
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Union[Sequence[int], int],
        method: Union[Method, str] = Method.SYMMETRIC,
        mode: NumpyPadModeSequence = NumpyPadMode.CONSTANT,
        verbose = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            spatial_size: the spatial size of output data after padding.
                If its components have non-positive values, the corresponding size of input image will be used.
            method: {``"symmetric"``, ``"end"``}
                Pad image symmetric on every side or only pad at the end sides. Defaults to ``"symmetric"``.
            mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
                ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                It also can be a sequence of string, each element corresponds to a key in ``keys``.

        """
        super().__init__(keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padder = SpatialPad(spatial_size, method)
        self.verbose = verbose

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        # print('[Pad] data keys', d.keys())
        for key, m in zip(self.keys, self.mode):
            # try:
                # print_tensor(f'[Pad]Pre: {key}', d[key])
            d[key] = self.padder(d[key], mode=m)
            d['img_meta_dict'][f'padshape'] = d[key].shape[-3:]
            if self.verbose: print_tensor(f'[Pad]Post: {key}', d[key])
            # except ValueError:
            #     print(f'[Pad] tensor error fn is', d['img_meta_dict']['filename_or_obj'])
        return d


class SpatialCropDJ(Transform):
    """
    General purpose cropper to produce sub-volume region of interest (ROI).
    It can support to crop ND spatial (channel-first) data.
    Either a spatial center and size must be provided, or alternatively,
    if center and size are not provided, the start and end coordinates of the ROI must be provided.
    """

    def __init__(
        self,
        roi_center: Optional[Sequence[int]] = None,
        roi_size: Optional[Sequence[int]] = None,
        roi_start: Optional[Sequence[int]] = None,
        roi_end: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Args:
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI.
            roi_start: voxel coordinates for start of the crop ROI.
            roi_end: voxel coordinates for end of the crop ROI.
        """

        if roi_center is not None and roi_size is not None:
            roi_center = np.asarray(roi_center, dtype=np.int16)
            roi_size = np.asarray(roi_size, dtype=np.int16)
            self.roi_start = roi_center - np.floor_divide(roi_size, 2)
            self.roi_end = self.roi_start + roi_size
            self.roi_size = roi_size
        else:
            if roi_start is None or roi_end is None:
                raise ValueError("Please specify either roi_center, roi_size or roi_start, roi_end.")
            self.roi_start = np.maximum(np.asarray(roi_start, dtype=np.int16), 0)
            self.roi_end = np.maximum(np.asarray(roi_end, dtype=np.int16), self.roi_start)
            self.roi_size = self.roi_end - self.roi_start


    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        # print('img', img.shape)
        image_shape = np.array(img.shape[1:])
        end_xyz_delta = self.roi_end - image_shape
        patch_shape = (img.shape[0], ) + tuple(self.roi_size)
        patch_image = np.full(patch_shape, img.min(), dtype = img.dtype)
        start_xyz_img = self.roi_start.clip(0) 
        end_xyz_img = self.roi_end.clip(0, image_shape)
        # slice4img = [slice(start_xyz_img[i], end_xyz_img[i]) for i in range(3)]

        start_xyz_patch = np.where(self.roi_start < 0, np.abs(self.roi_start),0)
        end_xyz_patch = np.where(end_xyz_delta > 0, self.roi_size - np.abs(end_xyz_delta), self.roi_size) #
        # slice4patch = [slice(start_xyz_patch[i], end_xyz_patch[i]) for i in range(3)]

        sd = min(len(self.roi_start), len(self.roi_end), len(img.shape[1:]))  # spatial dims
        slices4img = tuple([slice(None)] + [slice(s, e) for s, e in zip(start_xyz_img[:sd], end_xyz_img[:sd])])
        slices4patch = tuple([slice(None)] + [slice(s, e) for s, e in zip(start_xyz_patch[:sd], end_xyz_patch[:sd])])

        patch_image[slices4patch] = img[slices4img]

        return patch_image, slices4img, slices4patch

class CenterSpatialCrop_j(Transform):
    """
    Crop at the center of image with specified ROI size.

    Args:
        roi_size: the spatial size of the crop region e.g. [224,224,128]
            If its components have non-positive values, the corresponding size of input image will be used.
    """

    def __init__(self, roi_size: Union[Sequence[int], int], fixed_offset = (0, 0, 0)) -> None:
        self.roi_size = roi_size
        self.fixed_offset = fixed_offset

    def __call__(self, img: Union[np.ndarray, torch.Tensor], jitter_xyz = (0, 0, 0)) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        self.roi_size = fall_back_tuple(self.roi_size, img.shape[1:])
        # jitter_xyz = [randint(jitter) if jitter else 0 for _ in range(3)]
        img_shape = list(img.shape[1:])
        center = [min(s - self.roi_size[i]//2,
                    max(s // 2 + jitter_xyz[i] + self.fixed_offset[i], self.roi_size[i]//2)
                    )
                 for i, s in enumerate(img_shape)]
        cropper = SpatialCropDJ(roi_center=center, roi_size=self.roi_size)
        patch_img, slice4img, slice4patch = cropper(img)
        return patch_img, slice4img, slice4patch

class CenterSpatialCropDJ(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.CenterSpatialCrop_`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        roi_size: the size of the crop region e.g. [224,224,128]
            If its components have non-positive values, the corresponding size of input image will be used.
    """

    def __init__(self, keys: KeysCollection, roi_size: Union[Sequence[int], int], jitter = 0, 
                      fixed_offset = (0, 0, 0)) -> None:
        super().__init__(keys)
        self.jitter = jitter
        # self.fixed_offset = fixed_offset
        self.cropper = CenterSpatialCrop_j(roi_size, fixed_offset= fixed_offset)

    def __call__(self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        j4c = d['img_meta_dict'].get('jitter4center', None)
        jitter_xyz = [randint(j4c) if j4c else 0 for _ in range(3)]
        for key in self.keys:
            d[key], slice4img, slice4patch = self.cropper(d[key], jitter_xyz = jitter_xyz)
            d['img_meta_dict']['slice4img'] = slice4img
            d['img_meta_dict']['slice4patch'] = slice4patch
        return d

class RandCropByLabelBBoxRegiond(Randomizable, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandCropByLabelBBoxRegion`.
    Crop random fixed sized regions with the each label minimum bounding box region
    based on the Pos Neg Ratio. This is used for multi-label case. Each label will be cropped with same probability.
    And will return a list of dictionaries for all the cropped images.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        label_key: name of key for label image, this will be used for finding foreground/background.
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            If its components have non-positive values, the corresponding size of `data[label_key]` will be used.
        pyramid_scale: if use multiple pyramid level image outputs, set with multiple float type nums. e.g. [1.0, 2.0, 4.0]
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        pyramid_scale: Union[Sequence[float], float] = (1.0, ),
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        whole_image_as_bg: bool = True,
        percentile_00_5: Optional[float] = None,
        return_keys: KeysCollection = 'all',
        custom_center_kargs = {},
        data_info_csv = '',
        num_classes = 1,
        verbose = False
    ) -> None:
        super().__init__(keys)
        self.label_key = label_key
        self.spatial_size: Union[Tuple[int, ...], Sequence[int], int] = spatial_size
        self.pyramid_scale = pyramid_scale
        self.pyramid = True if len(pyramid_scale) > 1 else False
        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.centers: Optional[List[List[np.ndarray]]] = None
        self.whole_image_as_bg = whole_image_as_bg
        self.percentile_00_5 = percentile_00_5
        self.return_keys = return_keys
        self.custom_center_kargs = custom_center_kargs
        self.info_tb = None
        self.verbose = verbose
        self.num_classes = num_classes
        if data_info_csv: 
            if os.path.exists(data_info_csv):
                self.info_tb = pd.read_csv(data_info_csv, index_col = 'cid')
                print(self.info_tb.head())
    
    def custom_center_finding(self, label, base_value = None, jitter = 20, **kwargs):
        """
        
        """

        if isinstance(base_value, int): base_value = [base_value]
        # assert isinstance(base_value, (tuple, list))
        if base_value is None:
            VeinLoc = MaskFgLocator3d(label[0], image_center=True)
        else:
            # print('center base value', base_value)
            self.spatial_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
            mask_overlap_vein = (label == base_value[0]).astype(np.uint8)
            for bv in base_value[1:]:
                mask_overlap_vein += (label == bv).astype(np.uint8)
            VeinLoc = MaskFgLocator3d(mask_overlap_vein[0])

        center_dims = len(VeinLoc.obj_center_xyz)
        jitter_tuple = ensure_tuple_rep(jitter, center_dims)
        self.centers = [] # not use center
        for i, ix in enumerate(range(self.num_samples)):
            jitter_xyz = [0 if jitter ==0 else random.randrange(-jitter_tuple[d], jitter_tuple[d]) for d in range(center_dims)]
            # TODO: check if this jitter will be the same for different processes when using DDP
            center_jitter = [jitter_xyz[i] + VeinLoc.obj_center_xyz[i] for i in range(center_dims)]
            # print(f'[JITTER]', jitter_xyz, center_jitter)  #
            self.centers.append(center_jitter)
        
    def fetch_center_from_table(self, meta_data, jitter = 20):
        assert self.info_tb is not None
        # pdb.set_trace()
        fn = meta_data['filename_or_obj'].split(os.sep)[-1]
        case_name = '_'.join(fn.split('.')[0].split('_')[1:]) # [:2] or [1:]
        center_str = str(self.info_tb.loc[case_name, 'center_coord'])
        cstr = str(center_str).strip('"|[|]')
        ccord = [int(a) for a in cstr.split(',')]
        center_dims = len(ccord)
        jitter_tuple = ensure_tuple_rep(jitter, center_dims)
        self.centers = []
        for _ in enumerate(range(self.num_samples)):
            jitter_xyz = [random.randrange(-jitter_tuple[d], jitter_tuple[d]) for d in range(center_dims)]
            center_jitter = [jitter_xyz[i] + ccord[i] for i in range(center_dims)]
            self.centers.append(center_jitter)


    def __call__(self, data: Mapping[Hashable, Union[List[np.ndarray], np.ndarray, torch.Tensor]]
                ) -> List[Dict[Hashable, Union[np.ndarray, torch.Tensor]]]:
        d = dict(data)
        image = d[self.keys[0]]
        label = d[self.label_key]
        # print('[RandCrop] data keys', d.keys())
        # with Timer(print_tmpl='\tRandCrop finding center {:.3f} seconds'): #TODO
        fetch_center_func =  self.fetch_center_from_table
        if self.info_tb is not None: fetch_center_func(d['img_meta_dict'], self.custom_center_kargs['jitter'])
        elif self.custom_center_kargs: self.custom_center_finding(label, **self.custom_center_kargs)
        else: self.randomize(label, image)

        self.num_centers = len(self.centers)
        assert isinstance(self.spatial_size, tuple)
        assert self.centers is not None
        results: List[Dict[Hashable, Union[np.ndarray, torch.Tensor]]] = [dict() for _ in range(self.num_centers)]
        for key in data.keys():
            if key in self.keys and (self.return_keys == 'all' or key in self.return_keys):
                img = d[key]
                for i, center in enumerate(self.centers):
                    cropper =  SpatialCrop(roi_center=tuple(center), roi_size=self.spatial_size)
                    results[i][key] = cropper(img)
                    # if self.verbose: print_tensor(f'\n {i} {key}raw center {center}', img)
                    # if self.verbose: print_tensor(f' {i} {key} crop ', results[i][key])
            elif self.return_keys == 'all' or key in self.return_keys: # for meta_dict
                for i in range(self.num_centers):
                    results[i][key] = d[key]
        if self.verbose: print('Sampled data', results[0].keys())
        return results

class MaskFgLocator3d():

    def __init__(self, mask_3d, image_center = False):
        self.mask_3d = mask_3d
        self.dim = len(mask_3d.shape)
        self.image_center = image_center
        self.pos_idx_dim()

    @property
    def obj_loc_property(self):
        fg_coords = np.where(self.mask_3d > 0)
        
        self.pos_x_ixs = list(fg_coords[0]) #[z for z in range(self.mask_3d.shape[0]) if self.mask_3d[z].max() > 0]
        self.pos_y_ixs = list(fg_coords[1]) #[z for z in range(self.mask_3d.shape[1]) if self.mask_3d[:, z, ...].max() > 0]
        self.pos_z_ixs = list(fg_coords[2]) if self.dim == 3 else None  # [z for z in range(self.mask_3d.shape[2]) if self.mask_3d[..., z].max() > 0]

         # 3x4, row: xyz; col: coord_min, coord_max, coord_length, coord_center
        obj_loc_property = np.array(
                                    [self.range_in_list(list(fg_coords[i]), self.mask_3d.shape[i])
                                    # self.range_in_list(self.pos_y_ixs, self.mask_3d.shape[1]), 
                                    # self.range_in_list(self.pos_z_ixs, self.mask_3d.shape[2]) 
                                    for i in range(self.dim)
                                    ], dtype= np.int32)
        return obj_loc_property

    def range_in_list(self, ixs, slen):
        # range_in_list = lambda l: max(l) - min(l)
        ix_max = max(ixs) if len(ixs) > 0 else slen
        ix_min = min(ixs) if len(ixs) > 0 else 0
        try: 
            r = ix_max - ix_min
            c = (ix_max + ix_min)//2
        except ValueError:  
            r = slen
            c = slen//2
        return ix_min, ix_max, r, c

    def pos_idx_dim(self):

        if self.image_center:
            self.obj_shape_xyz = list(self.mask_3d.shape)
            self.obj_center_xyz = [s//2 for s in self.mask_3d.shape]
        else:
            self.obj_shape_xyz = self.obj_loc_property[:, 2]
            self.obj_center_xyz = self.obj_loc_property[:, 3]

    def extend_bbox(self,  extend = 0):
        new_min = np.array([max(a, 0) for a in (self.obj_loc_property[:, 0] - extend)])
        new_max = np.array([min(a, self.mask_3d.shape[i]) for i, a in enumerate(self.obj_loc_property[:, 1] + extend)])
        new_bbox = np.stack([new_min, new_max], axis = 1)
        return new_bbox

