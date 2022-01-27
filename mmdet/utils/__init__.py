# Copyright (c) OpenMMLab. All rights reserved.
from .collect_env import collect_env
from .logger import get_root_logger
from .misc import find_latest_checkpoint, LearningRateDecayOptimizerConstructor

print_tensor = lambda n, x: print(n, type(x), x.dtype, x.shape, x.min(), x.max())

__all__ = [
    'get_root_logger',
    'collect_env',
    'find_latest_checkpoint',
    'print_tensor', 'LearningRateDecayOptimizerConstructor'
]
