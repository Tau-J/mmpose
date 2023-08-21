# Copyright (c) OpenMMLab. All rights reserved.
from .coco_wholebody_dataset import CocoWholeBodyDataset
from .halpe_dataset import HalpeDataset
from .humanart21_dataset import HumanArt21Dataset
from .ubody2d_dataset import UBody2dDataset

__all__ = [
    'CocoWholeBodyDataset', 'HalpeDataset', 'UBody2dDataset',
    'HumanArt21Dataset'
]
