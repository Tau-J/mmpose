# Copyright (c) OpenMMLab. All rights reserved.
import glob

from mmpose.apis import MMPoseInferencer

inferencer = MMPoseInferencer(
    pose2d=
    'projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py',
    pose2d_weights=
    'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.pth'
)

datasets = {
    'freihand': {
        'train': {
            'src': '/nvme/data/pose/FreiHand/training/rgb',
            'ann_file': '/nvme/data/freihand/annotations/freihand_train.json',
        },
        'val': {
            'src': '/nvme/data/pose/FreiHand/evaluation/rgb',
            'ann_file': '/nvme/data/freihand/annotations/freihand_test.json',
        }
    }
}

data_scr = ['/nvme/data/pose/FreiHand/evaluation/rgb']

for src in data_scr:
    img_list = glob.glob(f'{src}/*.jpg')

    conducter = inferencer(img_list)

    for i, each in enumerate(conducter):
        print(img_list[i])
        # print(each)
        # break
        kpts = each['predictions'][0][0]['keypoints']
        scores = each['predictions'][0][0]['keypoint_scores']

        num = 21
        left_hand, right_hand = kpts[91:112], kpts[112:133]
        left_score, right_score = scores[91::112], scores[112:133]
        left_score, right_score = sum(left_score) / num, sum(right_score) / num

        print(f'left {left_score}, right {right_score}')
        if left_score > right_score:
            print('hand type: left')
        else:
            print('hand type: right')
        break
