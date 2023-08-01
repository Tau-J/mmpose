# Copyright (c) OpenMMLab. All rights reserved.

import os
from functools import partial

os.system('python -m mim install "mmcv>=2.0.0"')
os.system('python -m mim install "mmengine==0.8.2"')
os.system('python -m mim install "mmdet>=3.0.0"')
os.system('python -m mim install -e .')

# import mimetypes
from argparse import ArgumentParser

import cv2
import gradio as gr
import mmcv
import numpy as np

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0,
                      draw_heatmap=False):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


# input_type = 'image'


def predict(input, draw_heatmap=False, model_type='body', input_type='image'):
    """Visualize the demo images.

    Using mmdet to detect the human.
    """

    if model_type == 'face':
        det_config = 'demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py'  # noqa
        det_checkpoint = 'https://download.openmmlab.com/mmpose/mmdet_pretrained/yolo-x_8xb8-300e_coco-face_13274d7c.pth'  # noqa
        pose_config = 'projects/rtmpose/rtmpose/face_2d_keypoint/rtmpose-m_8xb64-120e_lapa-256x256.py'  # noqa
        pose_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth'  # noqa
    elif model_type == 'wholebody':
        det_config = 'projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py'  # noqa
        det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'  # noqa
        pose_config = 'projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py'  # noqa
        pose_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.pth'  # noqa
    else:
        det_config = 'projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py'  # noqa
        det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'  # noqa
        pose_config = 'projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-384x288.py'  # noqa
        pose_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.pth'  # noqa

    parser = ArgumentParser()
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
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
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=6,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=3,
        help='Link thickness for visualization')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    # build detector
    detector = init_detector(det_config, det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        pose_config,
        pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=draw_heatmap))))

    # input_type = 'image'
    # input_type = mimetypes.guess_type(args.input)[0].split('/')[0]
    print('input type', input_type, 'model_type', model_type)
    if input_type == 'image':
        # init visualizer
        from mmpose.registry import VISUALIZERS

        pose_estimator.cfg.visualizer.radius = args.radius
        pose_estimator.cfg.visualizer.alpha = args.alpha
        pose_estimator.cfg.visualizer.line_width = args.thickness
        visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)

        # the dataset_meta is loaded from the checkpoint and
        # then pass to the model in init_pose_estimator
        visualizer.set_dataset_meta(
            pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

        # inference
        _ = process_one_image(
            args,
            input[:, :, ::-1],
            detector,
            pose_estimator,
            visualizer,
            draw_heatmap=draw_heatmap)
        return visualizer.get_image()

    elif input_type == 'video':
        from mmpose.visualization import FastVisualizer

        visualizer = FastVisualizer(
            pose_estimator.dataset_meta,
            radius=args.radius,
            line_width=args.thickness,
            kpt_thr=args.kpt_thr)

        if draw_heatmap:
            # init Localvisualizer
            from mmpose.registry import VISUALIZERS

            pose_estimator.cfg.visualizer.radius = args.radius
            pose_estimator.cfg.visualizer.alpha = args.alpha
            pose_estimator.cfg.visualizer.line_width = args.thickness
            local_visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)

            # the dataset_meta is loaded from the checkpoint and
            # then pass to the model in init_pose_estimator
            local_visualizer.set_dataset_meta(
                pose_estimator.dataset_meta,
                skeleton_style=args.skeleton_style)

        cap = cv2.VideoCapture(input)

        video_writer = None
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            frame_idx += 1

            if not success:
                break

            # topdown pose estimation
            if draw_heatmap:
                pred_instances = process_one_image(
                    args,
                    frame,
                    detector,
                    pose_estimator,
                    local_visualizer,
                    0.001,
                    draw_heatmap=True)
            else:
                pred_instances = process_one_image(args, frame, detector,
                                                   pose_estimator)
                # visualization
                visualizer.draw_pose(frame, pred_instances)
                # cv2.imshow('MMPose Demo [Press ESC to Exit]', frame)

            # output videos
            if draw_heatmap:
                frame_vis = local_visualizer.get_image()
            else:
                frame_vis = frame.copy()[:, :, ::-1]

            output_file = 'test.mp4'
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # the size of the image with visualization may vary
                # depending on the presence of heatmaps
                video_writer = cv2.VideoWriter(
                    output_file,
                    fourcc,
                    25,  # saved fps
                    (frame_vis.shape[1], frame_vis.shape[0]))

            video_writer.write(mmcv.rgb2bgr(frame_vis))

        video_writer.release()
        cap.release()
        return output_file

    return None


# gr.Interface(
#     fn=predict,
#     inputs=[
#         gr.Image(type='numpy'),
#         gr.Image(type='webcam', )],
#     outputs=gr.Image(type='pil'),
#     examples=['tests/data/coco/000000000785.jpg']).launch()
news1 = '2023-8-1: We have supported [DWPose](https://arxiv.org/pdf/2307.15880.pdf) as the default `wholebody` model.'  # noqa
with gr.Blocks() as demo:

    with gr.Tab('Upload-Image'):
        input_img = gr.Image(type='numpy')
        button = gr.Button('Inference', variant='primary')
        hm = gr.Checkbox(label='draw-heatmap', info='Whether to draw heatmap')
        model_type = gr.Dropdown(['body', 'face', 'wholebody'],
                                 label='Keypoint Type',
                                 info='Body / Face / Wholebody')
        gr.Markdown('## News')
        gr.Markdown(news1)
        gr.Markdown('## Output')
        out_image = gr.Image(type='numpy')

        input_type = 'image'
        button.click(
            partial(predict, input_type=input_type),
            [input_img, hm, model_type], out_image)

    with gr.Tab('Webcam-Image'):
        input_img = gr.Image(source='webcam', type='numpy')
        button = gr.Button('Inference', variant='primary')
        hm = gr.Checkbox(label='draw-heatmap', info='Whether to draw heatmap')
        model_type = gr.Dropdown(['body', 'face', 'wholebody'],
                                 label='Keypoint Type',
                                 info='Body / Face / Wholebody')
        gr.Markdown('## News')
        gr.Markdown(news1)
        gr.Markdown('## Output')
        out_image = gr.Image(type='numpy')

        input_type = 'image'
        button.click(
            partial(predict, input_type=input_type),
            [input_img, hm, model_type], out_image)

    with gr.Tab('Upload-Video'):
        input_video = gr.Video(type='mp4')
        button = gr.Button('Inference', variant='primary')
        hm = gr.Checkbox(label='draw-heatmap', info='Whether to draw heatmap')
        model_type = gr.Dropdown(['body', 'face', 'wholebody'],
                                 label='Keypoint Type',
                                 info='Body / Face / Wholebody')
        gr.Markdown('## News')
        gr.Markdown(news1)
        gr.Markdown('## Output')
        out_video = gr.Video()

        input_type = 'video'
        button.click(
            partial(predict, input_type=input_type),
            [input_video, hm, model_type], out_video)

    with gr.Tab('Webcam-Video'):
        input_video = gr.Video(source='webcam', format='mp4')
        button = gr.Button('Inference', variant='primary')
        hm = gr.Checkbox(label='draw-heatmap', info='Whether to draw heatmap')
        model_type = gr.Dropdown(['body', 'face', 'wholebody'],
                                 label='Keypoint Type',
                                 info='Body / Face / Wholebody')
        gr.Markdown('## News')
        gr.Markdown(news1)
        gr.Markdown('## Output')
        out_video = gr.Video()

        input_type = 'video'
        button.click(
            partial(predict, input_type=input_type),
            [input_video, hm, model_type], out_video)

gr.close_all()
demo.queue()
demo.launch()
