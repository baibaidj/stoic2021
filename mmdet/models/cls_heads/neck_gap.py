import torch
from torch._C import Value
import torch.nn as nn

from ..builder import NECKS


@NECKS.register_module()
class GlobalAveragePooling(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling.
    We do not use `squeeze` as it will also remove the batch dimension
    when the tensor has a batch dimension of size 1, which can lead to
    unexpected errors.
    """

    def __init__(self, dim = 2):
        super(GlobalAveragePooling, self).__init__()

        if dim ==2:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        elif dim ==3:
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            raise ValueError(f'dim can only be 2 or 3 but got {dim}')

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, (tuple, list)):
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
