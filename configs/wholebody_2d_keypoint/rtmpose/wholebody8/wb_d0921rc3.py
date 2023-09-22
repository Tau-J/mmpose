_base_ = [
    './wb_d0917rc3.py'  # noqa: E501
]

# model settings
find_unused_parameters = False

# config settings
fea = True
logit = True

# method details
model = dict(
    _delete_=True,
    type='DWPoseDistiller',
    teacher_pretrained='/mnt/petrelfs/jiangtao/wb8_exp/wb_d0917rc2/'
    'best_coco-wholebody_AP_epoch_250.pth',  # noqa: E501 E251
    teacher_cfg='configs/wholebody_2d_keypoint/rtmpose/wholebody8/'
    'wb_d0917rc2.py',  # noqa: E501
    student_cfg='configs/wholebody_2d_keypoint/rtmpose/wholebody8/'
    'wb_d0917rc3.py',  # noqa: E501
    distill_cfg=[
        dict(methods=[
            dict(
                type='FeaLoss',
                name='loss_fea',
                use_this=fea,
                student_channels=1024,
                teacher_channels=1280,
                alpha_fea=0.00007,
            )
        ]),
        dict(methods=[
            dict(
                type='KDLoss',
                name='loss_logit',
                use_this=logit,
                weight=0.1,
            )
        ]),
    ],
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
)
optim_wrapper = dict(clip_grad=dict(max_norm=1., norm_type=2))
