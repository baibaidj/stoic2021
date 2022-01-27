# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv, torch

from ..builder import PIPELINES
from .compose import Compose
import copy

@PIPELINES.register_module()
class MultiScaleFlipAug:
    """Test-time augmentation with multiple scales and flipping.

    An example configuration is as followed:

    .. code-block::

        img_scale=[(1333, 400), (1333, 800)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]

    After MultiScaleFLipAug with above configuration, the results are wrapped
    into lists of the same length as followed:

    .. code-block::

        dict(
            img=[...],
            img_shape=[...],
            scale=[(1333, 400), (1333, 400), (1333, 800), (1333, 800)]
            flip=[False, True, False, True]
            ...
        )

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple] | None): Images scales for resizing.
        scale_factor (float | list[float] | None): Scale factors for resizing.
        flip (bool): Whether apply flip augmentation. Default: False.
        flip_direction (str | list[str]): Flip augmentation directions,
            options are "horizontal", "vertical" and "diagonal". If
            flip_direction is a list, multiple flip augmentations will be
            applied. It has no effect when flip == False. Default:
            "horizontal".
    """

    def __init__(self,
                 transforms,
                 img_scale=None,
                 scale_factor=None,
                 flip=False,
                 flip_direction='horizontal'):
        self.transforms = Compose(transforms)
        assert (img_scale is None) ^ (scale_factor is None), (
            'Must have but only one variable can be set')
        if img_scale is not None:
            self.img_scale = img_scale if isinstance(img_scale,
                                                     list) else [img_scale]
            self.scale_key = 'scale'
            assert mmcv.is_list_of(self.img_scale, tuple)
        else:
            self.img_scale = scale_factor if isinstance(
                scale_factor, list) else [scale_factor]
            self.scale_key = 'scale_factor'

        self.flip = flip
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmcv.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip
                and not any([t['type'] == 'RandomFlip' for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')

    def __call__(self, results):
        """Call function to apply test time augment transforms on results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
           dict[str: list]: The augmented data, where each value is wrapped
               into a list.
        """

        aug_data = []
        flip_args = [(False, None)]
        if self.flip:
            flip_args += [(True, direction)
                          for direction in self.flip_direction]
        for scale in self.img_scale:
            for flip, direction in flip_args:
                _results = results.copy()
                _results[self.scale_key] = scale
                _results['flip'] = flip
                _results['flip_direction'] = direction
                data = self.transforms(_results)
                aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip}, '
        repr_str += f'flip_direction={self.flip_direction})'
        return repr_str


@PIPELINES.register_module()
class MultiScaleFlipAug3D(object):
    """Test-time augmentation with multiple scales and flipping.

    An example configuration is as followed:

    .. code-block::

        img_scale=(2048, 1024),
        img_ratios=[0.5, 1.0],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]

    After MultiScaleFLipAug with above configuration, the results are wrapped
    into lists of the same length as followed:

    .. code-block::

        dict(
            img=[...],
            img_shape=[...],
            scale=[(1024, 512), (1024, 512), (2048, 1024), (2048, 1024)]
            flip=[False, True, False, True]
            ...
        )

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (None | tuple | list[tuple]): Images scales for resizing.
        img_ratios (float | list[float]): Image ratios for resizing
        flip (bool): Whether apply flip augmentation. Default: False.
        flip_direction (str | list[str]): Flip augmentation directions,
            options are 'diagonal', 'headfeet', 'all3axis'. If flip_direction is list,
            multiple flip augmentations will be applied.
            It has no effect when flip == False. Default: "horizontal".
    """

    def __init__(self,
                 transforms,
                 target_spacings = None,
                 img_scale = None,
                 flip=False,
                 flip_direction= ('diagonal', 'headfeat', 'all3axis'), #'diagonal', 'headfeet', 'all3axis'
                 label_mapping = None,
                 value4outlier = 0,
                 input_format = 'TCHW'):
        self.label_mapping = label_mapping
        self.value4outlier = value4outlier
        self.input_format = input_format
        self.transforms = Compose(transforms)
        if target_spacings is not None:
            self.target_spacings = target_spacings if isinstance(target_spacings, list) else [target_spacings]
        else:
            self.target_spacings = [None]

        self.flip = flip
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]

        assert mmcv.is_list_of(self.flip_direction, str)
        assert set(self.flip_direction) <= set(['diagonal', 'headfeet', 'all3axis', None]), f'Given directions are {self.flip_direction}'

        if not self.flip:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip
                and not any([t['type'] == 'FlipTTAd_' for t in transforms])):
            warnings.warn(
                'flip has no effect when FlipTTAd_ is not in transforms')

    def __call__(self, results):
        """Call function to apply test time augment transforms on results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
           dict[str: list]: The augmented data, where each value is wrapped
               into a list.
        """

        aug_data = []
        flip_aug = [False, True] if self.flip else [False]
        for spc in self.target_spacings:
            noflip = 0
            for flip in flip_aug:
                for direction in self.flip_direction:
                    # print(f'\t[TTA] TargetSpacing {spc}, Flip: {flip} {direction} ')
                    if not flip: noflip += 1
                    if not flip and noflip > 1: continue
                    _results = copy.deepcopy(results)
                    # only img_meta_dict is collected, so new key-value data is store there
                    # _results.setdefault('img_meta_dict', dict())
                    _results['img_meta_dict']['new_pixdim'] = spc
                    _results['img_meta_dict']['flip'] = flip
                    _results['img_meta_dict']['flip_direction'] = direction
                    data = self.transforms(_results)
                    if isinstance(data, list): aug_data.extend(data)
                    else: aug_data.append(data)

        # list of dict to dict of list
        aug_data_dict = {key: list() for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        
        # print(aug_data_dict['img_metas'])
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'target_spacings={self.target_spacings}, flip={self.flip})'
        repr_str += f'flip_direction={self.flip_direction}'
        repr_str += f'label_mapping={self.label_mapping}'
        return repr_str
