# Copyright (c) OpenMMLab. All rights reserved.
from torch.nn import functional as F


def interpolate_as(source, target, mode='bilinear', align_corners=False):
    """Interpolate the `source` to the shape of the `target`.

    The `source` must be a Tensor, but the `target` can be a Tensor or a
    np.ndarray with the shape (..., target_h, target_w).

    Args:
        source (Tensor): A 3D/4D Tensor with the shape (N, H, W) or
            (N, C, H, W).
        target (Tensor | np.ndarray): The interpolation target with the shape
            (..., target_h, target_w).
        mode (str): Algorithm used for interpolation. The options are the
            same as those in F.interpolate(). Default: ``'bilinear'``.
        align_corners (bool): The same as the argument in F.interpolate().

    Returns:
        Tensor: The interpolated source Tensor.
    """
    assert len(target.shape) >= 2

    def _interpolate_as(source, target, mode='bilinear', align_corners=False):
        """Interpolate the `source` (4D) to the shape of the `target`."""
        target_h, target_w = target.shape[-2:]
        source_h, source_w = source.shape[-2:]
        if target_h != source_h or target_w != source_w:
            source = F.interpolate(
                source,
                size=(target_h, target_w),
                mode=mode,
                align_corners=align_corners)
        return source

    if len(source.shape) == 3:
        source = source[:, None, :, :]
        source = _interpolate_as(source, target, mode, align_corners)
        return source[:, 0, :, :]
    else:
        return _interpolate_as(source, target, mode, align_corners)

import ipdb, torch

def nan_hook(self, inp, output):
    # print('input', type(inp), 'output', type(output))
    # ipdb.set_trace()

    def anything2list(out_raw):
        outs = []
        if isinstance(out_raw, (tuple, list)):
            for v in out_raw: 
                outs.extend(anything2list(v))
        elif isinstance(out_raw, dict):
            for k, v in out_raw.items():
                outs.extend(anything2list(v))
        elif isinstance(out_raw, torch.Tensor):
            return [out_raw]
        else:
            print(f'Unsetting type', type(out_raw))
        return outs

    outputs = anything2list(output)
    for i, out in enumerate(outputs):
        assert isinstance(out, torch.Tensor), (f'Out should be of type torch.Tensor but got {type(out)}')
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            print("[NaN output]@@In@@", self.__class__.__name__)
            # ipdb.set_trace()
            raise RuntimeError(f"Found NAN in output {i} at indices: ")
            #, nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])
