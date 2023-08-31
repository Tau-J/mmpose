# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from datetime import date

from tqdm import tqdm

os.system('pip install onnxconverter_common')

date_now = date.today().strftime('%Y%m%d')

det_cfg_src = '../mmpose-wb/projects/rtmpose/rtmdet/'
pose_cfg_src = '../mmpose-wb/projects/rtmpose/rtmpose/'

rtmpose_cfg = {
    # rtmdet
    'person-nano-320': f'{det_cfg_src}person/rtmdet_nano_320-8xb32_coco-person.py',
    'person-m-640': f'{det_cfg_src}person/rtmdet_m_640-8xb32_coco-person.py',
    'hand-nano-320': f'{det_cfg_src}hand/rtmdet_nano_320-8xb32_hand.py',
    # rtmpose
    'coco17-t-256': f'{pose_cfg_src}body_2d_keypoint/rtmpose-t_8xb256-420e_coco-256x192.py',
    'coco17-s-256': f'{pose_cfg_src}body_2d_keypoint/rtmpose-s_8xb256-420e_coco-256x192.py',
    'coco17-m-256': f'{pose_cfg_src}body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py',
    'coco17-l-256': f'{pose_cfg_src}body_2d_keypoint/rtmpose-l_8xb256-420e_coco-256x192.py',
    'coco17-m-384': f'{pose_cfg_src}body_2d_keypoint/rtmpose-m_8xb256-420e_coco-384x288.py',
    'coco17-l-384': f'{pose_cfg_src}body_2d_keypoint/rtmpose-l_8xb256-420e_coco-384x288.py',
    'coco17-x-384': f'{pose_cfg_src}body_2d_keypoint/rtmpose-x_8xb256-700e_coco-384x288.py',
    'halpe26-t-256':f'{pose_cfg_src}body_2d_keypoint/rtmpose-t_8xb1024-700e_body8-halpe26-256x192.py',
    'halpe26-s-256':f'{pose_cfg_src}body_2d_keypoint/rtmpose-s_8xb1024-700e_body8-halpe26-256x192.py',
    'halpe26-m-256':f'{pose_cfg_src}body_2d_keypoint/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py',
    'halpe26-l-256':f'{pose_cfg_src}body_2d_keypoint/rtmpose-l_8xb512-700e_body8-halpe26-256x192.py',
    'halpe26-m-384':f'{pose_cfg_src}body_2d_keypoint/rtmpose-m_8xb512-700e_body8-halpe26-384x288.py',
    'halpe26-l-384':f'{pose_cfg_src}body_2d_keypoint/rtmpose-l_8xb512-700e_body8-halpe26-384x288.py',
    'halpe26-x-384':f'{pose_cfg_src}body_2d_keypoint/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py',
    'coco133-t-256':f'{pose_cfg_src}wholebody_2d_keypoint/rtmpose-t_8xb64-270e_coco-wholebody-256x192.py',
    'coco133-s-256':f'{pose_cfg_src}wholebody_2d_keypoint/rtmpose-s_8xb64-270e_coco-wholebody-256x192.py',
    'coco133-m-256':f'{pose_cfg_src}wholebody_2d_keypoint/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py',
    'coco133-l-256':f'{pose_cfg_src}wholebody_2d_keypoint/rtmpose-l_8xb64-270e_coco-wholebody-256x192.py',
    'coco133-l-384':f'{pose_cfg_src}wholebody_2d_keypoint/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py',
    'coco133-x-384':f'{pose_cfg_src}wholebody_2d_keypoint/rtmpose-x_8xb32-270e_coco-wholebody-384x288.py',
    'ap10k-m-256': f'{pose_cfg_src}animal_2d_keypoint/rtmpose-m_8xb64-210e_ap10k-256x256.py',
    'lapa106-t-256': f'{pose_cfg_src}face_2d_keypoint/rtmpose-t_8xb256-120e_lapa-256x256.py',
    'lapa106-s-256': f'{pose_cfg_src}face_2d_keypoint/rtmpose-s_8xb256-120e_lapa-256x256.py',
    'lapa106-m-256': f'{pose_cfg_src}face_2d_keypoint/rtmpose-m_8xb256-120e_lapa-256x256.py',
    'coco21-m-256': f'{pose_cfg_src}hand_2d_keypoint/rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256.py'
}

model_cfg = [
    # det - person
    (
        rtmpose_cfg['person-nano-320'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'
    ),
    (
        rtmpose_cfg['person-m-640'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
    ),
    (
        rtmpose_cfg['hand-nano-320'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth'
    ),
    # body - 17kpt - body8 
    (
        rtmpose_cfg['coco17-t-256'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.pth'
    ),
    (
        rtmpose_cfg['coco17-s-256'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth'
    ),
    (
        rtmpose_cfg['coco17-m-256'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth'
    ),
    (
        rtmpose_cfg['coco17-l-256'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth'
    ),
    (
        rtmpose_cfg['coco17-m-384'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-384x288-65e718c4_20230504.pth'
    ),
    (
        rtmpose_cfg['coco17-l-384'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.pth'
    ),
    (
        rtmpose_cfg['coco17-x-384'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.pth'
    ),
    # body - 17kpt - humanart
    (
        rtmpose_cfg['coco17-t-256'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_8xb256-420e_humanart-256x192-60b68c98_20230612.pth'
    ),
    (
        rtmpose_cfg['coco17-s-256'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_8xb256-420e_humanart-256x192-5a3ac943_20230611.pth'
    ),
    (
        rtmpose_cfg['coco17-m-256'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_8xb256-420e_humanart-256x192-8430627b_20230611.pth'
    ),
    (
        rtmpose_cfg['coco17-l-256'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.pth'
    ),
    # body - 26kpt - body8
    (
        rtmpose_cfg['halpe26-t-256'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.pth'
    ),
    (
        rtmpose_cfg['halpe26-s-256'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.pth'
    ),
    (
        rtmpose_cfg['halpe26-m-256'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth'
    ),
    (
        rtmpose_cfg['halpe26-l-256'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.pth'
    ),
    (
        rtmpose_cfg['halpe26-m-384'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-384x288-89e6428b_20230605.pth'
    ),
    (
        rtmpose_cfg['halpe26-l-384'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-384x288-734182ce_20230605.pth'
    ),
    (
        rtmpose_cfg['halpe26-x-384'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.pth'
    ),
    # wholebody - 133kpt - dwpose
    (
        rtmpose_cfg['coco133-t-256'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_simcc-ucoco_dw-ucoco_270e-256x192-dcf277bf_20230728.pth'
    ),
    (
        rtmpose_cfg['coco133-s-256'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-ucoco_dw-ucoco_270e-256x192-3fd922c8_20230728.pth'
    ),
    (
        rtmpose_cfg['coco133-m-256'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.pth'
    ),
    (
        rtmpose_cfg['coco133-l-256'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-ucoco_dw-ucoco_270e-256x192-4d6dfc62_20230728.pth'
    ),
    (
        rtmpose_cfg['coco133-l-384'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.pth'
    ),
    # animal - 17kpt - ap10k
    (
        rtmpose_cfg['ap10k-m-256'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.pth'
    ),
    # face - 106kpt - face6
    (
        rtmpose_cfg['lapa106-t-256'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_simcc-face6_pt-in1k_120e-256x256-df79d9a5_20230529.pth'
    ),
    (
        rtmpose_cfg['lapa106-s-256'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-face6_pt-in1k_120e-256x256-d779fdef_20230529.pth'
    ),
    (
        rtmpose_cfg['lapa106-m-256'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth'
    ),
    # hand - 21kpt - hand5
    (
        rtmpose_cfg['coco21-m-256'],
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth'
    ),

]


print('check cfg exist...')
for key, cfg in rtmpose_cfg.items():
    assert osp.exists(cfg), f'{cfg} not exist!'
print('all cfg exist!')

# print('prepare ckpts...')
# for cfg, url in model_cfg:
#     os.system(f'wget -P ~/.cache/torch/hub/checkpoints/ {url}')

print('export model...')
results_list = []
for cfg, url in model_cfg:
    export_name = url.split('/')[-1]
    export_name = export_name[:-4]

    if 'rtmdet' in cfg:
        scripts = [
            'python tools/deploy.py',
            'configs/mmdet/detection/detection_onnxruntime_static.py',
            f'{cfg}',
            f'{url}',
            'demo/resources/human-pose.jpg',
            f'--work-dir {date_now}/rtmdet_onnx/{export_name}',
            '--device cpu',
            '--dump-info'
        ]
        results_list.append(f'{date_now}/rtmdet_onnx/{export_name}')
    else:
        scripts = [
            'python tools/deploy.py',
            'configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py',
            f'{cfg}',
            f'{url}',
            'demo/resources/human-pose.jpg',
            f'--work-dir {date_now}/rtmpose_onnx/{export_name}',
            '--device cpu',
            '--dump-info'
        ]
        results_list.append(f'{date_now}/rtmpose_onnx/{export_name}')

#     scripts = ' '.join(scripts)
#     os.system(scripts)
# print('all model exported!')

# print('onnx simpliy...')
# for result in tqdm(results_list):
#     os.system(f'onnxsim {result}/end2end.onnx {result}/end2end.onnx')

# print('zip...')
# for result in tqdm(results_list):
#     export_name = result.split('/')[-1]
#     os.system(f'zip -q -r {date_now}/{export_name}.zip {result}')

for result in results_list:
    if not osp.exists(f'{result}/end2end.onnx'):
        print(f'failed: {result}')