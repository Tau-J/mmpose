# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .wb_d1023rc2 import *  # noqa

from mmpose.models import DWPoseDistiller, PoseDataPreprocessor
from mmpose.models.losses import FeaLoss, KDLoss

# model settings
find_unused_parameters = False

# config settings
fea = True
logit = True

# method details
model = dict(
    type=DWPoseDistiller,
    teacher_pretrained='/mnt/petrelfs/jiangtao/wb8_exp/wb_d1030rc1/'
    'best_coco-wholebody_AP_epoch_200.pth',  # noqa: E501 E251
    teacher_cfg='configs/wholebody_2d_keypoint/rtmpose/wholebody8/'
    'wb_d1023rc1.py',  # noqa: E501
    student_cfg='configs/wholebody_2d_keypoint/rtmpose/wholebody8/'
    'wb_d1023rc2.py',  # noqa: E501
    distill_cfg=[
        dict(methods=[
            dict(
                type=FeaLoss,
                name='loss_fea',
                use_this=fea,
                student_channels=1024,
                teacher_channels=1280,
                alpha_fea=0.00007,
            )
        ]),
        dict(methods=[
            dict(
                type=KDLoss,
                name='loss_logit',
                use_this=logit,
                weight=0.1,
            )
        ]),
    ],
    data_preprocessor=dict(
        type=PoseDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
)

optim_wrapper.update(clip_grad=dict(max_norm=1., norm_type=2))  # noqa
