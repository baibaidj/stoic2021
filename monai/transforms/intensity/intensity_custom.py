

from monai.config import KeysCollection
from monai.transforms.transform import Randomizable, Transform, MapTransform
from monai.utils import ensure_tuple_size

import numpy as np
import torch

from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union


class NormalizeIntensityGPU(Transform):
    """
    Normalize input based on provided args with nnUNet method, crop intensity range within [0.5, 99.5],
    using calculated mean and std if not provided.
    This transform can normalize only non-zero values or entire image, and can also calculate
    mean and std on each channel separately.
    When `channel_wise` is True, the first dimension of `subtrahend` and `divisor` should
    be the number of image channels if they are not None.

    Args:
        subtrahend: the amount to subtract by (usually the mean).
        divisor: the amount to divide by (usually the standard deviation).
        percentile_99_5: the maximum intensity values.
        percentile_00_5: the minimum intensity values.
        nonzero: whether only normalize non-zero values.
        channel_wise: if using calculated mean and std, calculate on each channel separately
            or calculate on the entire image directly.
    """

    def __init__(
        self,
        subtrahend: Optional[Union[Sequence, float]] = None,
        divisor: Optional[Union[Sequence, float]] = None,
        percentile_99_5: Optional[Union[Sequence, float]] = None,
        percentile_00_5: Optional[Union[Sequence, float]] = None,
        nonzero: bool = False,
        channel_wise: bool = False,
    ) -> None:
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.percentile_99_5 = percentile_99_5
        self.percentile_00_5 = percentile_00_5
        self.nonzero = nonzero
        self.channel_wise = channel_wise

    def _normalize(self, img: Union[np.ndarray, torch.Tensor], 
                    sub=None, div=None, percentile_99_5=None, percentile_00_5=None
                    ) -> Union[np.ndarray, torch.Tensor]:
        if div == 0.0: div = 1.0
        if percentile_99_5 is not None and percentile_00_5 is not None:
            img = torch.clip(img, percentile_00_5, percentile_99_5)
        img = (img - sub) / div
        return img

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply the transform to `img`, assuming `img` is a channel-first array if `self.channel_wise` is True,
        """
        img = self._normalize(img, self.subtrahend, self.divisor, self.percentile_99_5, self.percentile_00_5)
        return img

class NormalizeIntensityGPUd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.NNUNetNormalizeIntensity`.
    This transform can normalize only non-zero values or entire image, and can also calculate
    mean and std on each channel separately.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        subtrahend: the amount to subtract by (usually the mean)
        divisor: the amount to divide by (usually the standard deviation)
        percentile_99_5: the maximum intensity values.
        percentile_00_5: the minimum intensity values.
        nonzero: whether only normalize non-zero values.
        channel_wise: if using calculated mean and std, calculate on each channel separately
            or calculate on the entire image directly.
    """

    def __init__(
            self,
            keys: KeysCollection,
            subtrahend: Optional[Union[Sequence, float]] = None,
            divisor: Optional[Union[Sequence, float]] = None,
            percentile_99_5: Optional[Union[Sequence, float]] = None,
            percentile_00_5: Optional[Union[Sequence, float]] = None,
            nonzero: bool = False,
            channel_wise: bool = False,
    ) -> None:
        super().__init__(keys)
        self.normalizer = NormalizeIntensityGPU(subtrahend, divisor, percentile_99_5, percentile_00_5, nonzero, channel_wise)

    def __call__(self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        # with Timer(print_tmpl='\tNORMIntensity {:.3f} seconds'): #TODO
        for key in self.keys:
            d[key] = self.normalizer(d[key])
        return d

from monai.utils.misc import dtype_numpy_to_torch
class RandGaussianNoised_(Randomizable, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandGaussianNoise_`.
    Add Gaussian noise to image. This transform assumes all the expected fields have same shape.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        prob: Probability to add Gaussian noise.
        mean: Mean or “centre” of the distribution.
        std: Standard deviation (spread) of distribution.
    """

    def __init__(
            self, keys: KeysCollection, prob: float = 0.1, mean: Union[Sequence[float], float] = 0.0, std: float = 0.1
    ) -> None:
        super().__init__(keys)
        self.prob = prob
        self.mean = ensure_tuple_size(mean, len(self.keys))
        self.std = std
        self._do_transform = False
        self._noise: Optional[np.ndarray] = None

    def randomize(self) -> None:
        self._do_transform = self.R.random() < self.prob
    
    def randnoise(self, im_shape: Sequence[int], device: torch.device):
        return torch.normal(mean=self.mean[0], std=self.R.uniform(0, self.std), size=im_shape).to(device)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        self.randomize()
        # with Timer(print_tmpl='\tRandNoise %s {:.3f} seconds'): #TODO
        if not self._do_transform:
            return d
        else:
            image_shape = d[self.keys[0]].shape  # image shape from the first data key
            rand_noise_tensor = self.randnoise(image_shape, d[self.keys[0]].device)
            for key in self.keys:
                dtype = dtype_numpy_to_torch(d[key].dtype) if isinstance(d[key], np.ndarray) else d[key].dtype
                d[key] = d[key] + rand_noise_tensor.to(dtype)
        return d