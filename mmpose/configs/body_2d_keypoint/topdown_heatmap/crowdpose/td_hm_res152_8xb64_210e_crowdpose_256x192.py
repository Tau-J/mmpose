# Copyright (c) OpenMMLab. All rights reserved.
if '_base_':
    from ...._base_.default_runtime import *

from mmengine.dataset.sampler import DefaultSampler
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from torch.optim.adam import Adam

from mmpose.codecs.msra_heatmap import MSRAHeatmap
from mmpose.datasets.datasets.body.crowdpose_dataset import CrowdPoseDataset
from mmpose.datasets.transforms.common_transforms import (GenerateTarget,
                                                          GetBBoxCenterScale,
                                                          RandomBBoxTransform,
                                                          RandomFlip,
                                                          RandomHalfBody)
from mmpose.datasets.transforms.formatting import PackPoseInputs
from mmpose.datasets.transforms.loading import LoadImage
from mmpose.datasets.transforms.topdown_transforms import TopdownAffine
from mmpose.evaluation.metrics.coco_metric import CocoMetric
from mmpose.models.backbones.resnet import ResNet
from mmpose.models.data_preprocessors.data_preprocessor import \
    PoseDataPreprocessor
from mmpose.models.heads.heatmap_heads.heatmap_head import HeatmapHead
from mmpose.models.losses.heatmap_loss import KeypointMSELoss
from mmpose.models.pose_estimators.topdown import TopdownPoseEstimator

# runtime
train_cfg.merge(dict(max_epochs=210, val_interval=10))

# optimizer
optim_wrapper = dict(optimizer=dict(
    type=Adam,
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(type=LinearLR, begin=0, end=500, start_factor=0.001,
         by_epoch=False),  # warm-up
    dict(
        type=MultiStepLR,
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks.merge(
    dict(checkpoint=dict(save_best='crowdpose/AP', rule='greater')))

# codec settings
codec = dict(
    type=MSRAHeatmap, input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# model settings
model = dict(
    type=TopdownPoseEstimator,
    data_preprocessor=dict(
        type=PoseDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type=ResNet,
        depth=152,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet152'),
    ),
    head=dict(
        type=HeatmapHead,
        in_channels=2048,
        out_channels=14,
        loss=dict(type=KeypointMSELoss, use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

# base dataset settings
dataset_type = 'CrowdPoseDataset'
data_mode = 'topdown'
data_root = 'data/crowdpose/'

# pipelines
train_pipeline = [
    dict(type=LoadImage),
    dict(type=GetBBoxCenterScale),
    dict(type=RandomFlip, direction='horizontal'),
    dict(type=RandomHalfBody),
    dict(type=RandomBBoxTransform),
    dict(type=TopdownAffine, input_size=codec['input_size']),
    dict(type=GenerateTarget, encoder=codec),
    dict(type=PackPoseInputs)
]
val_pipeline = [
    dict(type=LoadImage),
    dict(type=GetBBoxCenterScale),
    dict(type=TopdownAffine, input_size=codec['input_size']),
    dict(type=PackPoseInputs)
]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=CrowdPoseDataset,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/mmpose_crowdpose_trainval.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False, round_up=False),
    dataset=dict(
        type=CrowdPoseDataset,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/mmpose_crowdpose_test.json',
        bbox_file='data/crowdpose/annotations/det_for_crowd_test_0.1_0.5.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type=CocoMetric,
    ann_file=data_root + 'annotations/mmpose_crowdpose_test.json',
    use_area=False,
    iou_type='keypoints_crowd',
    prefix='crowdpose')
test_evaluator = val_evaluator
