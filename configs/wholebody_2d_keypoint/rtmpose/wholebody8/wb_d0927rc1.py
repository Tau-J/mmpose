_base_ = [
    './wb_d0917rc3.py'  # noqa: E501
]

# model settings
find_unused_parameters = True

# dis settings
second_dis = True

# config settings
logit = True

train_cfg = dict(max_epochs=60, val_interval=10)

self_dis_model_cfg = 'configs/wholebody_2d_keypoint/rtmpose/wholebody8/wb_d0917rc3.py'  # noqa: E501

# method details
model = dict(
    _delete_=True,
    type='DWPoseDistiller',
    two_dis=second_dis,
    teacher_pretrained='/mnt/petrelfs/jiangtao/wb8_exp/wb_d0917rc3/'
    's1.pth',  # noqa: E501
    teacher_cfg=self_dis_model_cfg,
    student_cfg=self_dis_model_cfg,
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
