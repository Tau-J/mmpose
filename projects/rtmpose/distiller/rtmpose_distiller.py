# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmpose.apis import init_model
from mmpose.evaluation.functional import simcc_pck_accuracy
from mmpose.models.pose_estimators import TopdownPoseEstimator
from mmpose.registry import MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, OptConfigType, OptMultiConfig,
                                 SampleList)


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
class PKDLoss(nn.Module):

    def __init__(
        self,
        use_target_weight=True,
        inter_loss_weight=1.0,
        intra_loss_weight=1.0,
        tau=1.0,
        loss_weight: float = 1.0,
        teacher_detach: bool = True,
    ):
        super(PKDLoss, self).__init__()
        self.inter_loss_weight = inter_loss_weight
        self.intra_loss_weight = intra_loss_weight
        self.tau = tau
        self.use_target_weight = use_target_weight

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
                                 coord_x_pred_T).mul(weight).mean()
            loss += self.pearson(coord_y_pred_S,
                                 coord_y_pred_T).mul(weight).mean()

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


def head_distill_forward(head, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:

    feats = feats[-1]

    feats = head.final_layer(feats)  # -> B, K, H, W

    # flatten the output heatmap
    feats = torch.flatten(feats, 2)

    feats = head.mlp(feats)  # -> B, K, hidden

    feats = head.gau(feats)

    pred_x = head.cls_x(feats)
    pred_y = head.cls_y(feats)

    return pred_x, pred_y, feats


@MODELS.register_module()
class RTMPoseDistiller(TopdownPoseEstimator):
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
        teacher_cfg (str, optional): The teacher model config file path.
            Defaults to ``None``
        teacher_ckpt (str, optional): The teacher model checkpoint file path.
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
                 metainfo: Optional[dict] = None,
                 teacher_cfg: str = None,
                 teacher_ckpt: str = None,
                 gau_distill: bool = False):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            metainfo=metainfo)

        self.gau_distill = gau_distill
        self.teacher = init_model(teacher_cfg, teacher_ckpt).eval()
        # self.distill_loss = RTMPoseDistillLoss(use_target_weight=True,
        #    tau=20.)
        self.distill_loss = PKDLoss(use_target_weight=True)

        # init tricks
        teacher_head_weights = self.teacher.head.state_dict()
        teacher_head_weights.popitem(0)  # remove final_layer.weight
        teacher_head_weights.popitem(0)  # remove final_layer.bias
        self.head.load_state_dict(teacher_head_weights, strict=False)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of losses.
        """
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

        feats = self.extract_feat(inputs)

        gau_loss = torch.tensor(0., device=gt_x.device)
        if self.gau_distill:
            pred_x, pred_y, pred_feats = head_distill_forward(self.head, feats)

            with torch.no_grad():
                self.teacher.eval()
                teacher_feats = self.teacher.extract_feat(inputs)
                teacher_x, teacher_y, teacher_feats = head_distill_forward(
                    self.teacher.head, teacher_feats)

            student_preds = (pred_x, pred_y)
            teacher_preds = (teacher_x, teacher_y)

            for idx in range(pred_feats.size(1)):
                gau_loss = gau_loss + self.distill_loss.pearson(
                    pred_feats[:, idx], teacher_feats[:, idx]).mean()
            gau_loss = gau_loss / pred_feats.size(1)
        else:
            student_preds = self.head.forward(feats)
            with torch.no_grad():
                teacher_preds = self.teacher.forward(inputs, data_samples)

        # calculate losses
        losses = dict()
        gt_loss = self.head.loss_module(student_preds, gt_simcc,
                                        keypoint_weights)
        distill_loss = self.distill_loss(student_preds, teacher_preds,
                                         keypoint_weights)
        gau_loss = gau_loss * 0.1

        losses.update(
            loss_kpt=gt_loss, distill_loss=distill_loss, gau_loss=gau_loss)

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
