from .cls_head import ClsHead
from .linear_head import LinearClsHead, LinearClsHeadCascade
from .neck_gap import GlobalAveragePooling
from .barlow_twin_head import BarlowTwinHead

__all__ = ['ClsHead', 'LinearClsHead', 
            'GlobalAveragePooling', 'LinearClsHeadCascade', 
            'BarlowTwinHead'
            ]
