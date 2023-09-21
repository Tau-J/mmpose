_base_ = [
    './wb_d0822rc2_rtm-m_256x192.py'  # noqa: E501
]

# model settings
find_unused_parameters = True

# dis settings
second_dis = True

# config settings
logit = True
max_epochs = 270

train_cfg = dict(max_epochs=60, val_interval=10)

base_lr = 4e-3

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),
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
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# method details
model = dict(
    _delete_=True,
    type='DWPoseDistiller',
    two_dis=second_dis,
    teacher_pretrained='/mnt/petrelfs/jiangtao/wb8_exp/'
    'wb_d0822rc2/s1_m.pth',  # noqa: E501
    teacher_cfg='configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/'
    'rtmpose-m_8xb64-270e_coco-wholebody-256x192.py',  # noqa: E501
    student_cfg='configs/wholebody_2d_keypoint/rtmpose/wholebody8/'
    'wb_d0822rc2_rtm-m_256x192.py',  # noqa: E501
    distill_cfg=[
        dict(methods=[
            dict(
                type='KDLoss',
                name='loss_logit',
                use_this=logit,
                weight=1,
            )
        ]),
    ],
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    train_cfg=train_cfg,
)

optim_wrapper = dict(clip_grad=dict(max_norm=1., norm_type=2))
