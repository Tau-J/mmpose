_base_ = ['../../../_base_/default_runtime.py']

# common setting
num_keypoints = 133
input_size = (192, 256)

# runtime
max_epochs = 270
stage2_num_epochs = 10
base_lr = 5e-4
train_batch_size = 64
val_batch_size = 32

train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=21)

custom_imports = dict(
    imports=['mmpose.engine.optim_wrappers.layer_decay_optim_wrapper'],
    allow_failed_imports=False)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=32,
        layer_decay_rate=0.85,
        custom_keys={
            'bias': dict(decay_multi=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
    ),
    constructor='LayerDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=1., norm_type=2),
)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
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
    type='SimCCLabel',
    input_size=input_size,
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# model settings
# load_from = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-ucoco_dw-ucoco_270e-256x192-4d6dfc62_20230728.pth'  # noqa

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmpretrain',
        type='VisionTransformer',
        arch='huge',
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.55,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/v1/'
            'body_2d_keypoint/topdown_heatmap/coco/'
            'td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth'
        ),
    ),
    head=dict(
        type='RTMCCHead',
        in_channels=1280,
        out_channels=num_keypoints,
        input_size=input_size,
        in_featuremap_size=tuple([s // 16 for s in input_size]),
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
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True))

# base dataset settings
dataset_type = 'CocoWholeBodyDataset'
data_mode = 'topdown'
data_root = 'data/'

# backend_args = dict(backend='local')
backend_args = dict(
    backend='petrel',
    path_mapping=dict({
        f'{data_root}detection/coco/':
        's254:s3://openmmlab/datasets/detection/coco/',
        f'{data_root}': 's3://openmmlab/datasets/',
    }))

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.5, 1.5], rotate_factor=90),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PhotometricDistortion'),
    dict(
        type='Albumentation',
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
                p=1.0),
        ]),
    dict(
        type='GenerateTarget',
        encoder=codec,
        use_dataset_keypoint_weights=True),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.5, 1.5],
        rotate_factor=90),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(
        type='Albumentation',
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
    dict(
        type='GenerateTarget',
        encoder=codec,
        use_dataset_keypoint_weights=True),
    dict(type='PackPoseInputs')
]

# mapping

aic_coco133 = [(0, 6), (1, 8), (2, 10), (3, 5), (4, 7), (5, 9), (6, 12),
               (7, 14), (8, 16), (9, 11), (10, 13), (11, 15)]

crowdpose_coco133 = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 11),
                     (7, 12), (8, 13), (9, 14), (10, 15), (11, 16)]

mpii_coco133 = [
    (0, 16),
    (1, 14),
    (2, 12),
    (3, 11),
    (4, 13),
    (5, 15),
    (8, 18),
    (9, 17),
    (10, 10),
    (11, 8),
    (12, 6),
    (13, 5),
    (14, 7),
    (15, 9),
]

jhmdb_coco133 = [
    (0, 18),
    (2, 17),
    (3, 6),
    (4, 5),
    (5, 12),
    (6, 11),
    (7, 8),
    (8, 7),
    (9, 14),
    (10, 13),
    (11, 10),
    (12, 9),
    (13, 16),
    (14, 15),
]

halpe_coco133 = [(i, i)
                 for i in range(17)] + [(20, 17), (21, 20), (22, 18), (23, 21),
                                        (24, 19),
                                        (25, 22)] + [(i, i - 3)
                                                     for i in range(26, 136)]

posetrack_coco133 = [
    (0, 0),
    (2, 17),
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

humanart_coco133 = [(i, i) for i in range(17)] + [(17, 99), (18, 120),
                                                  (19, 17), (20, 20)]

# train datasets
dataset_coco = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/coco_wholebody_train_v1.0.json',
    data_prefix=dict(img='detection/coco/train2017/'),
    pipeline=[],
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
            num_keypoints=num_keypoints,
            mapping=aic_coco133)
    ],
)

dataset_crowdpose = dict(
    type='CrowdPoseDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='crowdpose/annotations/mmpose_crowdpose_trainval.json',
    data_prefix=dict(img='pose/CrowdPose/images/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=crowdpose_coco133)
    ],
)

dataset_mpii = dict(
    type='MpiiDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='mpii/annotations/mpii_train.json',
    data_prefix=dict(img='pose/MPI/images/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=mpii_coco133)
    ],
)

dataset_jhmdb = dict(
    type='JhmdbDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='jhmdb/annotations/Sub1_train.json',
    data_prefix=dict(img='pose/JHMDB/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=jhmdb_coco133)
    ],
)

dataset_halpe = dict(
    type='HalpeDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='halpe/annotations/halpe_train_v1.json',
    data_prefix=dict(img='pose/Halpe/hico_20160224_det/images/train2015'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=halpe_coco133)
    ],
)

dataset_posetrack = dict(
    type='PoseTrack18Dataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='posetrack18/annotations/posetrack18_train.json',
    data_prefix=dict(img='pose/PoseChallenge2018/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=posetrack_coco133)
    ],
)

dataset_humanart = dict(
    type='HumanArt21Dataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='HumanArt/annotations/training_humanart.json',
    filter_cfg=dict(scenes=['real_human']),
    data_prefix=dict(img='pose/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=humanart_coco133)
    ])

ubody_scenes = [
    'Magic_show', 'Entertainment', 'ConductMusic', 'Online_class', 'TalkShow',
    'Speech', 'Fitness', 'Interview', 'Olympic', 'TVShow', 'Singing',
    'SignLanguage', 'Movie', 'LiveVlog', 'VideoConference'
]

ubody_datasets = []
for scene in ubody_scenes:
    each = dict(
        type='UBody2dDataset',
        data_root=data_root,
        data_mode=data_mode,
        ann_file=f'Ubody/annotations/{scene}/train_annotations.json',
        data_prefix=dict(img='pose/UBody/images/'),
        pipeline=[],
        sample_interval=10)
    ubody_datasets.append(each)

dataset_ubody = dict(
    type='CombinedDataset',
    metainfo=dict(from_file='configs/_base_/datasets/ubody2d.py'),
    datasets=ubody_datasets,
    pipeline=[],
    test_mode=False,
)

train_datasets = [
    dataset_coco,
    dataset_aic,
    dataset_crowdpose,
    dataset_mpii,
    dataset_jhmdb,
    dataset_halpe,
    dataset_posetrack,
    dataset_humanart,
    dataset_ubody,
]

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=10,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/coco_wholebody.py'),
        datasets=train_datasets,
        pipeline=train_pipeline,
        test_mode=False,
    ))

val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoWholeBodyDataset',
        ann_file='data/coco/annotations/coco_wholebody_val_v1.0.json',
        data_prefix=dict(img='data/detection/coco/val2017/'),
        pipeline=val_pipeline,
        bbox_file='data/coco/person_detection_results/'
        'COCO_val2017_detections_AP_H_56_person.json',
        test_mode=True))

test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(
        save_best='coco-wholebody/AP', rule='greater', max_keep_ckpts=1))
custom_hooks = [
    dict(
        type='EMAHook',
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
val_evaluator = dict(
    type='CocoWholeBodyMetric',
    ann_file='data/coco/annotations/coco_wholebody_val_v1.0.json')
test_evaluator = val_evaluator