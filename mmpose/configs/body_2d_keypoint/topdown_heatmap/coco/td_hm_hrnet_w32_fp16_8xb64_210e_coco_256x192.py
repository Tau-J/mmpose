# Copyright (c) OpenMMLab. All rights reserved.
if '_base_':
    from .td_hm_hrnet_w32_8xb64_210e_coco_256x192 import *
from mmengine.optim.optimizer.amp_optimizer_wrapper import AmpOptimWrapper

# fp16 settings
optim_wrapper.merge(dict(
    type=AmpOptimWrapper,
    loss_scale='dynamic',
))
