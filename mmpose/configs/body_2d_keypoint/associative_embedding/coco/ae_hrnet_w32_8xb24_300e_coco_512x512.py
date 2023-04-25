# Copyright (c) OpenMMLab. All rights reserved.
if '_base_':
    from ...._base_.default_runtime import *

from mmengine.dataset.sampler import DefaultSampler
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from torch.optim.adam import Adam

from mmpose.codecs.associative_embedding import AssociativeEmbedding
from mmpose.datasets.datasets.body.coco_dataset import CocoDataset
from mmpose.datasets.transforms.bottomup_transforms import BottomupResize
from mmpose.datasets.transforms.formatting import PackPoseInputs
from mmpose.datasets.transforms.loading import LoadImage
from mmpose.evaluation.metrics.coco_metric import CocoMetric
from mmpose.models.backbones.hrnet import HRNet
from mmpose.models.data_preprocessors.data_preprocessor import \
    PoseDataPreprocessor
from mmpose.models.heads.heatmap_heads.ae_head import AssociativeEmbeddingHead
from mmpose.models.losses.ae_loss import AssociativeEmbeddingLoss
from mmpose.models.losses.heatmap_loss import KeypointMSELoss
from mmpose.models.pose_estimators.bottomup import BottomupPoseEstimator

# runtime
train_cfg.merge(dict(max_epochs=300, val_interval=10))

# optimizer
optim_wrapper = dict(optimizer=dict(
    type=Adam,
    lr=1.5e-3,
))

# learning policy
param_scheduler = [
    dict(type=LinearLR, begin=0, end=500, start_factor=0.001,
         by_epoch=False),  # warm-up
    dict(
        type=MultiStepLR,
        begin=0,
        end=300,
        milestones=[200, 260],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=192)

# hooks
default_hooks.merge(
    dict(checkpoint=dict(save_best='coco/AP', rule='greater', interval=50)))

# codec settings
codec = dict(
    type=AssociativeEmbedding,
    input_size=(512, 512),
    heatmap_size=(128, 128),
    sigma=2,
    decode_keypoint_order=[
        0, 1, 2, 3, 4, 5, 6, 11, 12, 7, 8, 9, 10, 13, 14, 15, 16
    ],
    decode_max_instances=30)

# model settings
model = dict(
    type=BottomupPoseEstimator,
    data_preprocessor=dict(
        type=PoseDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type=HRNet,
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
            'pretrain_models/hrnet_w32-36af842e.pth'),
    ),
    head=dict(
        type=AssociativeEmbeddingHead,
        in_channels=32,
        num_keypoints=17,
        tag_dim=1,
        tag_per_keypoint=True,
        deconv_out_channels=None,
        keypoint_loss=dict(type=KeypointMSELoss, use_target_weight=True),
        tag_loss=dict(type=AssociativeEmbeddingLoss, loss_weight=0.001),
        # The heatmap will be resized to the input size before decoding
        # if ``restore_heatmap_size==True``
        decoder=dict(codec, heatmap_size=codec['input_size'])),
    test_cfg=dict(
        multiscale_test=False,
        flip_test=True,
        shift_heatmap=True,
        restore_heatmap_size=True,
        align_corners=False))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'bottomup'
data_root = 'data/coco/'

# pipelines
train_pipeline = []
val_pipeline = [
    dict(type=LoadImage),
    dict(
        type=BottomupResize,
        input_size=codec['input_size'],
        size_factor=32,
        resize_mode='expand'),
    dict(type=PackPoseInputs)
]

# data loaders
train_dataloader = dict(
    batch_size=24,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=CocoDataset,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False, round_up=False),
    dataset=dict(
        type=CocoDataset,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type=CocoMetric,
    ann_file=data_root + 'annotations/person_keypoints_val2017.json',
    nms_mode='none',
    score_mode='keypoint',
)
test_evaluator = val_evaluator
