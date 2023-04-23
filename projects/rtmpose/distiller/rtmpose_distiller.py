# Copyright (c) OpenMMLab. All rights reserved.
from itertools import zip_longest
from typing import Optional

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmpose.apis import init_model
from mmpose.evaluation.functional import simcc_pck_accuracy
from mmpose.models.pose_estimators.base import BasePoseEstimator
from mmpose.registry import MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1, keepdim=True),
                             b - b.mean(1, keepdim=True), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


@MODELS.register_module()
class DISTLoss(nn.Module):

    def __init__(
        self,
        inter_loss_weight=1.0,
        intra_loss_weight=1.0,
        tau=1.0,
        loss_weight: float = 1.0,
        teacher_detach: bool = True,
    ):
        super(DISTLoss, self).__init__()
        self.inter_loss_weight = inter_loss_weight
        self.intra_loss_weight = intra_loss_weight
        self.tau = tau
        self.use_target_weight = True

        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach

    def pearson(self, logits_S, logits_T: torch.Tensor):
        if self.teacher_detach:
            logits_T = logits_T.detach()
        y_s = (logits_S / self.tau).softmax(dim=1)
        y_t = (logits_T / self.tau).softmax(dim=1)
        inter_loss = self.tau**2 * inter_class_relation(y_s, y_t)
        intra_loss = self.tau**2 * intra_class_relation(y_s, y_t)
        kd_loss = self.inter_loss_weight * inter_loss + self.intra_loss_weight * intra_loss  # noqa
        return kd_loss * self.loss_weight

    def forward(self, pred_simcc_S, pred_simcc_T, target_weight_S):
        output_x_S, output_y_S = pred_simcc_S
        output_x_T, output_y_T = pred_simcc_T
        num_joints = output_x_S.size(1)
        loss = 0

        for idx in range(num_joints):
            coord_x_pred_S = output_x_S[:, idx].squeeze()
            coord_y_pred_S = output_y_S[:, idx].squeeze()
            coord_x_pred_T = output_x_T[:, idx].squeeze()
            coord_y_pred_T = output_y_T[:, idx].squeeze()

            if self.use_target_weight:
                weight = target_weight_S[:, idx].squeeze()
            else:
                weight = 1.

            loss += self.pearson(coord_x_pred_S,
                                 coord_x_pred_T).mul(weight).sum()
            loss += self.pearson(coord_y_pred_S,
                                 coord_y_pred_T).mul(weight).sum()

        return loss * self.loss_weight / num_joints


def kl_div(preds_S, preds_T, tau: float = 1.0):
    """Calculate the KL divergence between `preds_S` and `preds_T`.

    Args:
        preds_S (torch.Tensor): The student model prediction with shape (N, C).
        preds_T (torch.Tensor): The teacher model prediction with shape (N, C).
        tau (float): Temperature coefficient.
    """
    softmax_pred_T = F.softmax(preds_T / tau, dim=1)
    logsoftmax_preds_S = F.log_softmax(preds_S / tau, dim=1)
    loss = (tau**2) * F.kl_div(
        logsoftmax_preds_S, softmax_pred_T, reduction='none')
    loss = loss.sum(-1)
    return loss


@MODELS.register_module()
class RTMPoseDistillLoss(nn.Module):
    """A measure of how one probability distribution Q is different from a
    second, reference probability distribution P.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        reduction (str): Specifies the reduction to apply to the loss:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied,
            ``'batchmean'``: the sum of the output will be divided by
                the batchsize,
            ``'sum'``: the output will be summed,
            ``'mean'``: the output will be divided by the number of
                elements in the output.
            Default: ``'batchmean'``
        loss_weight (float): Weight of loss. Defaults to 1.0.
        teacher_detach (bool): Whether to detach the teacher model prediction.
            Will set to ``'False'`` in some data-free distillation algorithms.
            Defaults to True.
    """

    def __init__(self,
                 use_target_weight: bool = True,
                 tau: float = 1.0,
                 loss_weight: float = 1.0):
        super(RTMPoseDistillLoss, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.use_target_weight = use_target_weight

    def forward(self, pred_simcc_S, pred_simcc_T, target_weight_S):
        output_x_S, output_y_S = pred_simcc_S
        output_x_T, output_y_T = pred_simcc_T
        num_joints = output_x_S.size(1)
        loss = 0

        for idx in range(num_joints):
            coord_x_pred_S = output_x_S[:, idx].squeeze()
            coord_y_pred_S = output_y_S[:, idx].squeeze()
            coord_x_pred_T = output_x_T[:, idx].squeeze()
            coord_y_pred_T = output_y_T[:, idx].squeeze()

            if self.use_target_weight:
                weight = target_weight_S[:, idx].squeeze()
            else:
                weight = 1.

            loss += kl_div(
                coord_x_pred_S, coord_x_pred_T,
                tau=self.tau).mul(weight).sum()
            loss += kl_div(
                coord_y_pred_S, coord_y_pred_T,
                tau=self.tau).mul(weight).sum()

        return loss * self.loss_weight / num_joints


@MODELS.register_module()
class RTMPoseDistiller(BasePoseEstimator):
    """Distiller for top-down pose estimators.

    Args:
        backbone (dict): The backbone config
        neck (dict, optional): The neck config. Defaults to ``None``
        head (dict, optional): The head config. Defaults to ``None``
        train_cfg (dict, optional): The runtime config for training process.
            Defaults to ``None``
        test_cfg (dict, optional): The runtime config for testing process.
            Defaults to ``None``
        data_preprocessor (dict, optional): The data preprocessing config to
            build the instance of :class:`BaseDataPreprocessor`. Defaults to
            ``None``
        init_cfg (dict, optional): The config to control the initialization.
            Defaults to ``None``
        metainfo (dict): Meta information for dataset, such as keypoints
            definition and properties. If set, the metainfo of the input data
            batch will be overridden. For more details, please refer to
            https://mmpose.readthedocs.io/en/latest/user_guides/
            prepare_datasets.html#create-a-custom-dataset-info-
            config-file-for-the-dataset. Defaults to ``None``
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 teacher_cfg: str = None,
                 teacher_ckpt: str = None,
                 metainfo: Optional[dict] = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            metainfo=metainfo)

        self.teacher = init_model(teacher_cfg, teacher_ckpt).eval()
        # self.distill_loss = RTMPoseDistillLoss(use_target_weight=True,
        #    tau=20.)
        self.distill_loss = DISTLoss()

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of losses.
        """
        feats = self.extract_feat(inputs)
        student_preds = self.head.forward(feats)

        with torch.no_grad():
            teacher_preds = self.teacher.forward(inputs, data_samples)

        losses = dict()

        gt_x = torch.cat(
            [d.gt_instance_labels.keypoint_x_labels for d in data_samples],
            dim=0)
        gt_y = torch.cat(
            [d.gt_instance_labels.keypoint_y_labels for d in data_samples],
            dim=0)
        keypoint_weights = torch.cat(
            [d.gt_instance_labels.keypoint_weights for d in data_samples],
            dim=0,
        )

        gt_simcc = (gt_x, gt_y)

        # calculate losses
        losses = dict()
        gt_loss = self.head.loss_module(student_preds, gt_simcc,
                                        keypoint_weights)
        distill_loss = self.distill_loss(student_preds, teacher_preds,
                                         keypoint_weights)
        loss = gt_loss + distill_loss

        losses.update(
            loss_kpt=loss, gt_loss=gt_loss, distill_loss=distill_loss)

        # calculate accuracy
        _, avg_acc, _ = simcc_pck_accuracy(
            output=to_numpy(student_preds),
            target=to_numpy(gt_simcc),
            simcc_split_ratio=self.head.simcc_split_ratio,
            mask=to_numpy(keypoint_weights) > 0,
        )

        acc_pose = torch.tensor(avg_acc, device=gt_x.device)
        losses.update(acc_pose=acc_pose)

        return losses

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W)
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples

        Returns:
            list[:obj:`PoseDataSample`]: The pose estimation results of the
            input images. The return value is `PoseDataSample` instances with
            ``pred_instances`` and ``pred_fields``(optional) field , and
            ``pred_instances`` usually contains the following keys:

                - keypoints (Tensor): predicted keypoint coordinates in shape
                    (num_instances, K, D) where K is the keypoint number and D
                    is the keypoint dimension
                - keypoint_scores (Tensor): predicted keypoint scores in shape
                    (num_instances, K)
        """
        assert self.with_head, (
            'The model must have head to perform prediction.')

        if self.test_cfg.get('flip_test', False):
            _feats = self.extract_feat(inputs)
            _feats_flip = self.extract_feat(inputs.flip(-1))
            feats = [_feats, _feats_flip]
        else:
            feats = self.extract_feat(inputs)

        preds = self.head.predict(feats, data_samples, test_cfg=self.test_cfg)

        if isinstance(preds, tuple):
            batch_pred_instances, batch_pred_fields = preds
        else:
            batch_pred_instances = preds
            batch_pred_fields = None

        results = self.add_pred_to_datasample(batch_pred_instances,
                                              batch_pred_fields, data_samples)

        return results

    def add_pred_to_datasample(self, batch_pred_instances: InstanceList,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples: SampleList) -> SampleList:
        """Add predictions into data samples.

        Args:
            batch_pred_instances (List[InstanceData]): The predicted instances
                of the input data batch
            batch_pred_fields (List[PixelData], optional): The predicted
                fields (e.g. heatmaps) of the input batch
            batch_data_samples (List[PoseDataSample]): The input data batch

        Returns:
            List[PoseDataSample]: A list of data samples where the predictions
            are stored in the ``pred_instances`` field of each data sample.
        """
        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []
        output_keypoint_indices = self.test_cfg.get('output_keypoint_indices',
                                                    None)

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            gt_instances = data_sample.gt_instances

            # convert keypoint coordinates from input space to image space
            bbox_centers = gt_instances.bbox_centers
            bbox_scales = gt_instances.bbox_scales
            input_size = data_sample.metainfo['input_size']

            pred_instances.keypoints = pred_instances.keypoints / input_size \
                * bbox_scales + bbox_centers - 0.5 * bbox_scales

            if output_keypoint_indices is not None:
                # select output keypoints with given indices
                num_keypoints = pred_instances.keypoints.shape[1]
                for key, value in pred_instances.all_items():
                    if key.startswith('keypoint'):
                        pred_instances.set_field(
                            value[:, output_keypoint_indices], key)

            # add bbox information into pred_instances
            pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                if output_keypoint_indices is not None:
                    # select output heatmap channels with keypoint indices
                    # when the number of heatmap channel matches num_keypoints
                    for key, value in pred_fields.all_items():
                        if value.shape[0] != num_keypoints:
                            continue
                        pred_fields.set_field(value[output_keypoint_indices],
                                              key)
                data_sample.pred_fields = pred_fields

        return batch_data_samples
