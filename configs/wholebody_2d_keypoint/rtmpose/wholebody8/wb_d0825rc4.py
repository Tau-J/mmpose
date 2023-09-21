_base_ = [
    './wb_d0822rc2_rtm-m_256x192.py'  # noqa: E501
]

max_epochs = 60

# model settings
find_unused_parameters = True

# dis settings
second_dis = True

# config settings
logit = True

train_cfg = dict(max_epochs=max_epochs, val_interval=10)

# method details
model = dict(
    _delete_=True,
    type='DWPoseDistiller',
    two_dis=second_dis,
    teacher_pretrained='/mnt/petrelfs/jiangtao/wb8_exp/'
    'wb_d0822rc2/s1_m.pth',  # noqa: E501
    teacher_cfg='configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/'
    'rtmpose-m_8xb64-270e_coco-wholebody-256x192.py',  # noqa: E501
    student_cfg='configs/wholebody_2d_keypoint/dwpose/ubody/s1_dis/'
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

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]
