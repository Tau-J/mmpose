# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import os.path as osp

import mmengine
from mmengine.fileio import dump

from mmpose.apis import MMPoseInferencer
from mmpose.datasets.datasets.hand import (FreiHandDataset, OneHand10KDataset,
                                           Rhd2DDataset)
from mmpose.registry import DATASETS

# import shutil

inferencer = MMPoseInferencer(
    pose2d='projects/rtmpose/rtmpose/wholebody_2d_keypoint/'
    'rtmpose-l_8xb32-270e_coco-wholebody-384x288.py',
    pose2d_weights='https://download.openmmlab.com/mmpose/v1/'
    'projects/rtmposev1/'
    'rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.pth')

data_root = '/nvme/data/'
data_mode = 'topdown'

datasets = {
    'freihand': {
        'train':
        dict(
            type=FreiHandDataset,
            data_root=data_root,
            data_mode=data_mode,
            ann_file='freihand/annotations/freihand_train.json',
            data_prefix=dict(img='pose/FreiHand/'),
            pipeline=[],
        ),
        'test':
        dict(
            type=FreiHandDataset,
            data_root=data_root,
            data_mode=data_mode,
            ann_file='freihand/annotations/freihand_test.json',
            data_prefix=dict(img='pose/FreiHand/'),
            pipeline=[],
        )
    },
    'onehand10k': {
        'train':
        dict(
            type=OneHand10KDataset,
            data_root=data_root,
            data_mode=data_mode,
            ann_file='onehand10k/annotations/onehand10k_train.json',
            data_prefix=dict(img='pose/OneHand10K/'),
            pipeline=[],
        ),
        'test':
        dict(
            type=OneHand10KDataset,
            data_root=data_root,
            data_mode=data_mode,
            ann_file='onehand10k/annotations/onehand10k_test.json',
            data_prefix=dict(img='pose/OneHand10K/'),
            pipeline=[],
        )
    },
    'rhd2d': {
        'train':
        dict(
            type=Rhd2DDataset,
            data_root=data_root,
            data_mode=data_mode,
            ann_file='rhd/annotations/rhd_train.json',
            data_prefix=dict(img='pose/RHD/'),
            pipeline=[],
        ),
        'test':
        dict(
            type=Rhd2DDataset,
            data_root=data_root,
            data_mode=data_mode,
            ann_file='rhd/annotations/rhd_test.json',
            data_prefix=dict(img='pose/RHD/'),
            pipeline=[],
        )
    }
}

for name, dataset in datasets.items():
    for tv in ['train', 'test']:

        image_infos = []
        annotations = []
        img_ids = []
        ann_ids = []
        instance_list = []
        each_dataset = DATASETS.build(dataset[tv])

        for img_id in each_dataset.coco.getImgIds():
            if img_id % each_dataset.sample_interval != 0:
                continue
            img = each_dataset.coco.loadImgs(img_id)[0]
            img.update({
                'img_id':
                img_id,
                'img_path':
                osp.join(each_dataset.data_prefix['img'], img['file_name']),
            })

            if img['img_id'] not in img_ids:
                image_info = dict(
                    id=img['img_id'],
                    file_name=img['file_name'],
                    width=img['width'],
                    height=img['height'],
                )
                if 'crowdIndex' in img:
                    image_info['crowdIndex'] = img['crowdIndex']

                image_infos.append(image_info)
                img_ids.append(img['img_id'])

            ann_ids = each_dataset.coco.getAnnIds(imgIds=img_id)
            for ann in each_dataset.coco.loadAnns(ann_ids):

                img_path = img['img_path']

                # print(img_path)
                res = next(inferencer(img_path))

                kpts = res['predictions'][0][0]['keypoints']
                scores = res['predictions'][0][0]['keypoint_scores']

                num = 21
                left_hand, right_hand = kpts[91:112], kpts[112:133]
                left_score, right_score = scores[91::112], scores[112:133]
                left_score = sum(left_score) / num
                right_score = sum(right_score) / num
                # print(f'left {left_score}, right {right_score}')
                if left_score > right_score:
                    # print('hand type: left')
                    hand_type = 'left'
                else:
                    # print('hand type: right')
                    hand_type = 'right'
                # print('-'*20)
                # mmengine.mkdir_or_exist('./hand_reco/')
                # img_name = img_path.split('/')[-1]
                # shutil.copyfile(img_path, f'./hand_reco/left-{left_score}_right-{right_score}_{hand_type}_{img_name}')  # noqa
                hand_type_valid = 1

                annotation = dict(
                    id=ann['id'],
                    image_id=ann['image_id'],
                    category_id=ann['category_id'],
                    bbox=ann['bbox'],
                    keypoints=ann['keypoints'],
                    hand_type=hand_type,
                    hand_type_valid=hand_type_valid,
                    iscrowd=ann['iscrowd'],
                    area=ann['area'],
                )

                if 'num_keypoints' in ann:
                    annotation['num_keypoints'] = ann['num_keypoints']

                annotations.append(annotation)
                ann_ids.append(ann['id'])

        info = dict(
            date_created=str(datetime.datetime.now()),
            description=f'{name} json file converted by MMPose.')
        coco_json = dict(
            info=info,
            images=image_infos,
            categories=each_dataset._metainfo['CLASSES'],
            licenses=None,
            annotations=annotations,
        )
        mmengine.mkdir_or_exist('./hand_datasets/')
        converted_json_path = f'./hand_datasets/{name}_{tv}_with_handtype.json'
        dump(coco_json, converted_json_path, sort_keys=True, indent=4)
