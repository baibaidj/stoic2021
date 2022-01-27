"""
Copyright 2020 KEYA Medical Algorithm Team
Version: 1.0
"""

import torch
from .dataloader import DataLoader
from monai.transforms import Compose
import numpy as np


class DataLoaderGPU(DataLoader):
    """
    Split batch to patch and process patch by aug_gpu

    batch struct:
    {
        'image': torch.tensor((batch, channel, ...)),
        'label': torch.tensor((batch, channel, ...)),
        'xxx_meta_dict': {'key1': [patch1, patch2,...], 
                          'key2': [patch1, patch2,...], 
                          ...},
        ...
    }
    """
    def __init__(self, aug_gpu: Compose, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug_gpu = aug_gpu

    def split_batch(self, data):
        keys = list(data.keys())
        size = None
        for k in keys:
            if isinstance(data[k], dict):
                continue
            size = len(data[k])

        result = []
        for b in range(size):
            res = {}
            for k in data:
                if isinstance(data[k], dict):
                    res[k] = {j: data[k][j][b] for j in data[k]}
                else:
                    res[k] = data[k][b]
            result.append(res)

        return result

    def merge_batch(self, data):
        size = len(data)
        keys = list(data[0].keys())
        result = {
            k: {o: []
                for o in data[0][k]} if isinstance(data[0][k], dict) else []
            for k in keys
        }
        for k in keys:
            for b in range(size):
                if isinstance(data[b][k], dict):
                    for o in data[b][k]:
                        result[k][o].append(data[b][k][o])
                else:
                    result[k].append(data[b][k])
        return result

    def loop_batch(self, data):
        data = [self.aug_gpu(p) for p in self.split_batch(data)]
        data = self.merge_batch(data)
        for k in data:
            if isinstance(data[k], dict):
                continue
            if isinstance(data[k][0], (torch.Tensor, np.ndarray)):
                data[k] = torch.stack(data[k], 0)
        return data

    def __iter__(self):
        for data in super().__iter__():
            if len(self.aug_gpu.transforms) != 0:
                data = self.loop_batch(data)
            yield data




class MultiEpochsDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)