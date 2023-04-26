# Copyright (c) OpenMMLab. All rights reserved.
if '_base_':
    from ...._base_.default_runtime import *

from mmengine.dataset.sampler import DefaultSampler
from mmengine.hooks.ema_hook import EMAHook
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.nn.modules.activation import SiLU
from torch.optim.adamw import AdamW

from mmpose.codecs.simcc_label import SimCCLabel
from mmpose.datasets.datasets.body.mpii_dataset import MpiiDataset
from mmpose.datasets.transforms.common_transforms import (Albumentation,
                                                          GenerateTarget,
                                                          GetBBoxCenterScale,
                                                          RandomBBoxTransform,
                                                          RandomFlip,
                                                          RandomHalfBody)
from mmpose.datasets.transforms.formatting import PackPoseInputs
from mmpose.datasets.transforms.loading import LoadImage
from mmpose.datasets.transforms.topdown_transforms import TopdownAffine
from mmpose.evaluation.metrics.keypoint_2d_metrics import MpiiPCKAccuracy
from mmpose.models.data_preprocessors.data_preprocessor import \
    PoseDataPreprocessor
from mmpose.models.heads.coord_cls_heads.rtmcc_head import RTMCCHead
from mmpose.models.losses.classification_loss import KLDiscretLoss
from mmpose.models.pose_estimators.topdown import TopdownPoseEstimator

# runtime
max_epochs = 210
stage2_num_epochs = 30
base_lr = 4e-3

train_cfg.merge(dict(max_epochs=max_epochs, val_interval=10))

# optimizer
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=AdamW, lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning policy
param_scheduler = [
    dict(
        type=LinearLR, start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(
        # use cosine lr from 210 to 420 epoch
        type=CosineAnnealingLR,
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=1024)

# codec settings
codec = dict(
    type=SimCCLabel,
    input_size=(256, 256),
    sigma=(5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# model settings
model = dict(
    type=TopdownPoseEstimator,
    data_preprocessor=dict(
        type=PoseDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type=SiLU),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
            'rtmposev1/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth'  # noqa
        )),
    head=dict(
        type=RTMCCHead,
        in_channels=768,
        out_channels=16,
        input_size=codec['input_size'],
        in_featuremap_size=(8, 8),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type=KLDiscretLoss,
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True))

# base dataset settings
dataset_type = 'MpiiDataset'
data_mode = 'topdown'
data_root = 'data/mpii/'

backend_args.merge(dict(backend='local'))
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         f'{data_root}': 's3://openmmlab/datasets/pose/MPI/',
#         f'{data_root}': 's3://openmmlab/datasets/pose/MPI/'
#     }))

# pipelines
train_pipeline = [
    dict(type=LoadImage, backend_args=backend_args),
    dict(type=GetBBoxCenterScale),
    dict(type=RandomFlip, direction='horizontal'),
    dict(type=RandomHalfBody),
    dict(type=RandomBBoxTransform, scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type=TopdownAffine, input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type=Albumentation,
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.),
        ]),
    dict(type=GenerateTarget, encoder=codec),
    dict(type=PackPoseInputs)
]
val_pipeline = [
    dict(type=LoadImage, backend_args=backend_args),
    dict(type=GetBBoxCenterScale),
    dict(type=TopdownAffine, input_size=codec['input_size']),
    dict(type=PackPoseInputs)
]

train_pipeline_stage2 = [
    dict(type=LoadImage, backend_args=backend_args),
    dict(type=GetBBoxCenterScale),
    dict(type=RandomFlip, direction='horizontal'),
    dict(type=RandomHalfBody),
    dict(
        type=RandomBBoxTransform,
        shift_factor=0.,
        scale_factor=[0.75, 1.25],
        rotate_factor=60),
    dict(type=TopdownAffine, input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type=Albumentation,
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type=GenerateTarget, encoder=codec),
    dict(type=PackPoseInputs)
]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=MpiiDataset,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/mpii_train.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False, round_up=False),
    dataset=dict(
        type=MpiiDataset,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/mpii_val.json',
        headbox_file=f'{data_root}/annotations/mpii_gt_val.mat',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# hooks
default_hooks.merge(
    dict(checkpoint=dict(save_best='PCK', rule='greater', max_keep_ckpts=1)))

custom_hooks = [
    dict(
        type=EMAHook,
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# evaluators
val_evaluator = dict(type=MpiiPCKAccuracy)
test_evaluator = val_evaluator
