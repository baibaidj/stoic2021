# Copyright (c) OpenMMLab. All rights reserved.
import collections

from mmcv.utils import build_from_cfg

from ..builder import PIPELINES
from typing import Any, Callable

@PIPELINES.register_module()
class Compose(object):
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = apply_transform(t, data)#t(data)
            if data is None:
                return None
        return data

    def _select_transform(self, data, outer_ix = 0, inner_ix = (0, 2, 3), force2key = 'img'):
        for ii in inner_ix:
            t = self.transforms[outer_ix].transforms.transforms[ii]
            t.keys = [force2key]
            data = apply_transform(t, data)
            if data is None: return
        return data
                
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


def apply_transform(transform: Callable, data, map_items: bool = True):
    """
    Transform `data` with `transform`.
    If `data` is a list or tuple and `map_data` is True, each item of `data` will be transformed
    and this method returns a list of outcomes.
    otherwise transform will be applied once with `data` as the argument.

    Args:
        transform: a callable to be used to transform `data`
        data: an object to be transformed.
        map_items: whether to apply transform to each item in `data`,
            if `data` is a list or tuple. Defaults to True.

    Raises:
        Exception: When ``transform`` raises an exception.

    """
    try:
        if isinstance(data, (list, tuple)) and map_items:
            # num_data = len(data)
            # print(f'data is list {num_data}', 'tranformed by', transform)
            return [transform(item) for item in data]
        return transform(data)
    except Exception as e:
        raise RuntimeError(f"applying transform {transform}") from e

