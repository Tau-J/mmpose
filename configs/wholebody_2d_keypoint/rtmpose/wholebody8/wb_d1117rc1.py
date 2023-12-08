from mmengine.config import read_base

with read_base():
    from .wb_d1023rc5 import *  # noqa

from mmpose.models import DWPoseDistiller, PoseDataPreprocessor
from mmpose.models.losses import KDLoss

# model settings
find_unused_parameters = True

# dis settings
second_dis = True

# config settings
logit = True

train_cfg.update(max_epochs=60, val_interval=10)  # noqa

# method details
model = dict(
    type=DWPoseDistiller,
    two_dis=second_dis,
    teacher_pretrained='/mnt/petrelfs/jiangtao/wb8_exp/wb_d1109rc2/'
    's1.pth',  # noqa: E501 E251
    teacher_cfg='configs/wholebody_2d_keypoint/rtmpose/wholebody8/'
    'wb_d1023rc5.py',  # noqa: E501
    student_cfg='configs/wholebody_2d_keypoint/rtmpose/wholebody8/'
    'wb_d1023rc5.py',  # noqa: E501
    distill_cfg=[
        dict(methods=[
            dict(
                type=KDLoss,
                name='loss_logit',
                use_this=logit,
                weight=1,
            )
        ]),
    ],
    data_preprocessor=dict(
        type=PoseDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    train_cfg=train_cfg,  # noqa
)

optim_wrapper.update(clip_grad=dict(max_norm=1., norm_type=2))  # noqa
