# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List, Tuple

import numpy as np
from mmengine.utils import check_file_exist
from xtcocotools.coco import COCO

from mmpose.registry import DATASETS
from mmpose.structures.bbox import bbox_xywh2xyxy
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class HalpeHandDataset(BaseCocoStyleDataset):
    """HalpeDataset for hand pose estimation.

    'https://github.com/Fang-Haoshu/Halpe-FullBody'

    Halpe Hand keypoints::

        0: 'wrist',
        1: 'thumb1',
        2: 'thumb2',
        3: 'thumb3',
        4: 'thumb4',
        5: 'forefinger1',
        6: 'forefinger2',
        7: 'forefinger3',
        8: 'forefinger4',
        9: 'middle_finger1',
        10: 'middle_finger2',
        11: 'middle_finger3',
        12: 'middle_finger4',
        13: 'ring_finger1',
        14: 'ring_finger2',
        15: 'ring_finger3',
        16: 'ring_finger4',
        17: 'pinky_finger1',
        18: 'pinky_finger2',
        19: 'pinky_finger3',
        20: 'pinky_finger4'

    Args:
        ann_file (str): Annotation file path. Default: ''.
        bbox_file (str, optional): Detection result file path. If
            ``bbox_file`` is set, detected bboxes loaded from this file will
            be used instead of ground-truth bboxes. This setting is only for
            evaluation, i.e., ignored when ``test_mode`` is ``False``.
            Default: ``None``.
        data_mode (str): Specifies the mode of data samples: ``'topdown'`` or
            ``'bottomup'``. In ``'topdown'`` mode, each data sample contains
            one instance; while in ``'bottomup'`` mode, each data sample
            contains all instances in a image. Default: ``'topdown'``
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Default: ``None``.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Default: ``None``.
        data_prefix (dict, optional): Prefix for training data. Default:
            ``dict(img=None, ann=None)``.
        filter_cfg (dict, optional): Config for filter data. Default: `None`.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Default: ``None`` which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy.
            Default: ``True``.
        pipeline (list, optional): Processing pipeline. Default: [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Default: ``False``.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Default: ``False``.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Default: 1000.
    """

    METAINFO: dict = dict(from_file='configs/_base_/datasets/halpe_hand.py')

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        """Load data from annotations in COCO format."""

        def get_bbox(keypoints):
            """Get bbox from keypoints."""
            x1, y1, _ = np.amin(keypoints, axis=0)
            x2, y2, _ = np.amax(keypoints, axis=0)
            w, h = x2 - x1, y2 - y1
            return [x1, y1, w, h]

        check_file_exist(self.ann_file)

        coco = COCO(self.ann_file)
        instance_list = []
        image_list = []
        id = 0

        for img_id in coco.getImgIds():
            img = coco.loadImgs(img_id)[0]

            img.update({
                'img_id':
                img_id,
                'img_path':
                osp.join(self.data_prefix['img'], img['file_name']),
            })
            image_list.append(img)

            ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                keypoints = np.array(ann['keypoints']).reshape(-1, 3)
                lefthand_kpts = keypoints[-42:-21, :]
                righthand_kpts = keypoints[-21:, :]

                left_mask = lefthand_kpts[:, 2] > 0
                lefthand_box = get_bbox(lefthand_kpts[left_mask, :])
                right_mask = righthand_kpts[:, 2] > 0
                righthand_box = get_bbox(righthand_kpts[right_mask, :])
                t_ann = {
                    'lefthand_kpts': lefthand_kpts,
                    'righthand_kpts': righthand_kpts,
                    'lefthand_valid': np.max(lefthand_kpts) > 0,
                    'righthand_valid': np.max(righthand_kpts) > 0,
                    'lefthand_box': lefthand_box,
                    'righthand_box': righthand_box,
                }
                for hand_type in ['left', 'right']:
                    # filter invalid hand annotations, there might be two
                    # valid instances (left and right hand) in one image
                    if t_ann[f'{hand_type}hand_valid']:
                        bbox_xywh = np.array(
                            t_ann[f'{hand_type}hand_box'],
                            dtype=np.float32).reshape(1, 4)

                        bbox = bbox_xywh2xyxy(bbox_xywh)

                        _keypoints = np.array(
                            t_ann[f'{hand_type}hand_kpts'],
                            dtype=np.float32).reshape(1, -1, 3)
                        keypoints = _keypoints[..., :2]
                        keypoints_visible = np.minimum(1, _keypoints[..., 2])

                        num_keypoints = np.count_nonzero(keypoints.max(axis=2))

                        hand_type = ann.get('hand_type', None)
                        hand_type_valid = ann.get('hand_type_valid', 0)

                        instance_info = {
                            'img_id': ann['image_id'],
                            'img_path': img['img_path'],
                            'bbox': bbox,
                            'bbox_score': np.ones(1, dtype=np.float32),
                            'num_keypoints': num_keypoints,
                            'keypoints': keypoints,
                            'keypoints_visible': keypoints_visible,
                            'hand_type': self.encode_handtype(hand_type),
                            'hand_type_valid': hand_type_valid,
                            'iscrowd': ann['iscrowd'],
                            'id': id,
                        }
                        instance_list.append(instance_info)
                        id = id + 1

        instance_list = sorted(instance_list, key=lambda x: x['id'])
        return instance_list, image_list

    @staticmethod
    def encode_handtype(hand_type):
        if hand_type == 'right':
            return np.array([[1, 0]], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([[0, 1]], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([[1, 1]], dtype=np.float32)
        else:
            return np.array([[-1, -1]], dtype=np.float32)
