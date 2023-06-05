_base_ = './rtmpose-s_8xb256-210e_body8-256x192.py'

_base_.model._scope_ = 'mmpose'

teacher_ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth'  # noqa

# codec settings
codec_tea = dict(
    type='SimCCLabel',
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)
teacher = dict(
    _scope_='mmpose',
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
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')),
    head=dict(
        type='RTMHead',
        in_channels=768,
        out_channels=17,
        input_size=codec_tea['input_size'],
        in_featuremap_size=(6, 8),
        simcc_split_ratio=codec_tea['simcc_split_ratio'],
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
        decoder=codec_tea),
    test_cfg=dict(flip_test=True))

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    data_preprocessor=dict(
        type='mmpose.PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    architecture=_base_.model,
    teacher=teacher,
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            logits=dict(type='ModuleInputs', source='head.loss_module')),
        teacher_recorders=dict(
            logits=dict(type='ModuleInputs', source='head.loss_module')),
        distill_losses=dict(
            loss_rtm_pose=dict(
                type='RTMPoseDistillLoss',
                loss_weight=1,
                use_target_weight=False,
                tau=20)),
        loss_forward_mappings=dict(
            loss_rtm_pose=dict(
                pred_simcc_S=dict(
                    from_student=True, recorder='logits', data_idx=0),
                pred_simcc_T=dict(
                    from_student=False, recorder='logits', data_idx=0),
                target_weight_S=dict(
                    from_student=True, recorder='logits', data_idx=2)))))

find_unused_parameters = True

interval = 10
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=interval, max_keep_ckpts=3))

optim_wrapper = dict(clip_grad=dict(max_norm=5, norm_type=2))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=_base_.max_epochs - _base_.stage2_num_epochs,
        switch_pipeline=_base_.train_pipeline_stage2),
    dict(
        type='mmrazor.DistillationLossDetachHook',
        detach_epoch=_base_.max_epochs - _base_.stage2_num_epochs)
]
