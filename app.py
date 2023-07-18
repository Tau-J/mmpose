# Copyright (c) OpenMMLab. All rights reserved.


import os

os.system('python -m mim install "mmcv>=2.0.1"')
os.system('python -m mim install mmengine')
os.system('python -m mim install "mmdet>=3.1.0"')
os.system('python -m mim install -e .')

from argparse import ArgumentParser
from typing import Dict

import cv2
import gradio as gr
import mmengine

from mmpose.apis.inferencers import MMPoseInferencer, get_model_aliases


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs',
        type=str,
        default=None,
        help='Input image/video path or folder path.')
    parser.add_argument(
        '--pose2d',
        type=str,
        default=None,
        help='Pretrained 2D pose estimation algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--pose2d-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected pose model. '
        'If it is not specified and "pose2d" is a model name of metafile, '
        'the weights will be loaded from metafile.')
    parser.add_argument(
        '--pose3d',
        type=str,
        default=None,
        help='Pretrained 3D pose estimation algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--pose3d-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected pose model. '
        'If it is not specified and "pose3d" is a model name of metafile, '
        'the weights will be loaded from metafile.')
    parser.add_argument(
        '--det-model',
        type=str,
        default=None,
        help='Config path or alias of detection model.')
    parser.add_argument(
        '--det-weights',
        type=str,
        default=None,
        help='Path to the checkpoints of detection model.')
    parser.add_argument(
        '--det-cat-ids',
        type=int,
        nargs='+',
        default=0,
        help='Category id for detection model.')
    parser.add_argument(
        '--scope',
        type=str,
        default='mmpose',
        help='Scope where modules are defined.')
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device used for inference. '
        'If not specified, the available device will be automatically used.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image/video in a popup window.')
    parser.add_argument(
        '--draw-bbox',
        action='store_true',
        help='Whether to draw the bounding boxes.')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Whether to draw the predicted heatmaps.')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--use-oks-tracking',
        action='store_true',
        help='Whether to use OKS as similarity in tracking')
    parser.add_argument(
        '--norm-pose-2d',
        action='store_true',
        help='Scale the bbox (along with the 2D pose) to the average bbox '
        'scale of the dataset, and move the bbox (along with the 2D pose) to '
        'the average bbox center of the dataset. This is useful when bbox '
        'is small, especially in multi-person scenarios.')
    parser.add_argument(
        '--rebase-keypoint-height',
        action='store_true',
        help='Rebase the predicted 3D pose so its lowest keypoint has a '
        'height of 0 (landing on the ground). This is useful for '
        'visualization when the model do not predict the global position '
        'of the 3D pose.')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization.')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization.')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--black-background',
        action='store_true',
        help='Plot predictions on a black image')
    parser.add_argument(
        '--vis-out-dir',
        type=str,
        default='',
        help='Directory for saving visualized results.')
    parser.add_argument(
        '--pred-out-dir',
        type=str,
        default='',
        help='Directory for saving inference results.')
    parser.add_argument(
        '--show-alias',
        action='store_true',
        help='Display all the available model aliases.')

    call_args = vars(parser.parse_args())

    init_kws = [
        'pose2d', 'pose2d_weights', 'scope', 'device', 'det_model',
        'det_weights', 'det_cat_ids', 'pose3d', 'pose3d_weights'
    ]
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    diaplay_alias = call_args.pop('show_alias')

    return init_args, call_args, diaplay_alias


def display_model_aliases(model_aliases: Dict[str, str]) -> None:
    """Display the available model aliases and their corresponding model
    names."""
    aliases = list(model_aliases.keys())
    max_alias_length = max(map(len, aliases))
    print(f'{"ALIAS".ljust(max_alias_length+2)}MODEL_NAME')
    for alias in sorted(aliases):
        print(f'{alias.ljust(max_alias_length+2)}{model_aliases[alias]}')


def predict(input_img, input_type):
    init_args, call_args, _ = parse_args()

    init_args['pose3d'] = 'configs/body_3d_keypoint/pose_lift/h36m/pose-lift_motionbert-243frm_8xb32-120e_h36m.py'  # noqa
    init_args['pose3d_weights'] = 'https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/motionbert_ft_h36m-d80af323_20230531.pth'  # noqa
    
    call_args['inputs'] = input_img
    call_args['rebase_keypoint_height'] = True

    if input_type == 'image':
        inferencer = MMPoseInferencer(**init_args)
        call_args['vis_out_dir'] = './pose3d_results.jpg'
        gen = inferencer(**call_args)
        _ = next(gen)
        img = cv2.imread('./pose3d_results.jpg')
        os.remove('./pose3d_results.jpg')
        return img
    else:
        inferencer = MMPoseInferencer(**init_args)
        call_args['vis_out_dir'] = './pose3d_results.mp4'
        for _ in inferencer(**call_args):
            pass
        return './pose3d_results.mp4'


mmengine.mkdir_or_exist(os.path.join('./', 'resources'))

os.system(
    f'wget -O resources/tom.mp4 https://download.openmmlab.com/mmpose/v1/projects/just_dance/tom.mp4'  # noqa
)
with gr.Blocks() as demo:

    with gr.Tab('Upload-Image'):
        input_img = gr.Image(type='numpy')
        button = gr.Button('Inference', variant='primary')

        gr.Markdown('## Output')
        out_image = gr.Image(type='numpy')

        button.click(predict, [input_img, 'image'], out_image)

        gr.Examples([
            'tests/data/coco/000000000785.jpg'
        ])

    with gr.Tab('Upload-Video'):
        input_video = gr.Video(type='mp4')
        button = gr.Button('Inference', variant='primary')

        gr.Markdown('## Output')
        out_video = gr.Video()
        
        input_type = 'video'
        button.click(predict, [input_video, 'video'], out_video)

        gr.Examples([
            'resources/tom.mp4'
        ])


gr.close_all()
demo.queue()
demo.launch()
