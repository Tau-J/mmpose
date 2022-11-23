_base_ = ['../../_base_/default_runtime.py']

# runtime
max_epochs = 210
stage2_num_epochs = 30
base_lr = 4e-3

train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=3407)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=1., norm_type=2),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.01,
        begin=max_epochs // 3,
        end=max_epochs,
        T_max=2 * max_epochs // 3,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=1024)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=True)

# keypoint mappings
keypoint_mapping_coco = [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (7, 7),
    (8, 8),
    (9, 9),
    (10, 10),
    (11, 11),
    (12, 12),
    (13, 13),
    (14, 14),
    (15, 15),
    (16, 16),
]

keypoint_mapping_aic = [
    (0, 6),
    (1, 8),
    (2, 10),
    (3, 5),
    (4, 7),
    (5, 9),
    (6, 12),
    (7, 14),
    (8, 16),
    (9, 11),
    (10, 13),
    (11, 15),
    (12, 17),
    (13, 18),
]

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
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
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='/mnt/petrelfs/jiangtao/pretrained_models/'
            'cspnext-m_coco_256x192.pth')),
    head=dict(
        type='RTMHead',
        in_channels=768,
        out_channels=19,
        input_size=codec['input_size'],
        in_featuremap_size=(6, 8),
        simcc_split_ratio=codec['simcc_split_ratio'],
        use_hilbert_flatten=True,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            shift=True,
            shift_type='time',
            dropout_rate=0.3,
            drop_path=0.,
            act_fn='StarReLU',
            use_rel_bias=False,
        ),
        use_coord_token=True,
        axis_align=True,
        num_self_attn=1,
        use_cross_attn=True,
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=False,
        output_keypoint_indices=[
            target for _, target in keypoint_mapping_coco
        ]))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/'

file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        f'{data_root}': 's3://openmmlab/datasets/',
        f'{data_root}': 's3://openmmlab/datasets/'
    }))

# pipelines
train_pipeline = [
    dict(type='LoadImage', file_client_args=file_client_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.5, 1.5], rotate_factor=80),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PhotometricDistortion'),
    # dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            # dict(type='ToGray', p=0.01),
            # dict(type='CLAHE', p=0.01),
            # dict(
            #     type='CoarseDropout',
            #     max_holes=1,
            #     max_height=0.4,
            #     max_width=0.4,
            #     min_holes=1,
            #     min_height=0.2,
            #     min_width=0.2,
            #     p=0.7),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', file_client_args=file_client_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImage', file_client_args=file_client_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        scale_factor=[0.75, 1.25],
        rotate_factor=60),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            # dict(type='ToGray', p=0.01),
            # dict(type='CLAHE', p=0.01),
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
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# train datasets
dataset_coco = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/person_keypoints_train2017.json',
    data_prefix=dict(img='detection/coco/train2017/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=19,
            mapping=keypoint_mapping_coco)
    ],
)

dataset_aic = dict(
    type='AicDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='aic/annotations/aic_train.json',
    data_prefix=dict(img='pose/ai_challenge/ai_challenger_keypoint'
                     '_train_20170902/keypoint_train_images_20170902/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=19,
            mapping=keypoint_mapping_aic)
    ],
)

# data loaders
train_dataloader = dict(
    batch_size=128 * 2,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/coco_aic.py'),
        datasets=[dataset_coco, dataset_aic],
        pipeline=train_pipeline,
        test_mode=False,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='coco/annotations/person_keypoints_val2017.json',
        bbox_file='data/coco/person_detection_results/'
        'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='detection/coco/val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='coco/AP', rule='greater', max_keep_ckpts=1))

custom_hooks = [
    # dict(
    #     type='EMAHook',
    #     ema_type='ExpMomentumEMA',
    #     momentum=0.0002,
    #     update_buffers=True,
    #     priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'coco/annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator