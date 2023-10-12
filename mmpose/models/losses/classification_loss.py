# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS


@MODELS.register_module()
class BCELoss(nn.Module):
    """Binary Cross Entropy loss.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of the loss. Default: 1.0.
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            before output. Defaults to False.
    """

    def __init__(self,
                 use_target_weight=False,
                 loss_weight=1.,
                 reduction='mean',
                 use_sigmoid=False):
        super().__init__()

        assert reduction in ('mean', 'sum', 'none'), f'the argument ' \
            f'`reduction` should be either \'mean\', \'sum\' or \'none\', ' \
            f'but got {reduction}'

        self.reduction = reduction
        self.use_sigmoid = use_sigmoid
        criterion = F.binary_cross_entropy if use_sigmoid \
            else F.binary_cross_entropy_with_logits
        self.criterion = partial(criterion, reduction='none')
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K]): Output classification.
            target (torch.Tensor[N, K]): Target classification.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """

        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output, target)
            if target_weight.dim() == 1:
                target_weight = target_weight[:, None]
            loss = (loss * target_weight)
        else:
            loss = self.criterion(output, target)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss * self.loss_weight


@MODELS.register_module()
class JSDiscretLoss(nn.Module):
    """Discrete JS Divergence loss for DSNT with Gaussian Heatmap.

    Modified from `the official implementation
    <https://github.com/anibali/dsntnn/blob/master/dsntnn/__init__.py>`_.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        size_average (bool): Option to average the loss by the batch_size.
    """

    def __init__(
        self,
        use_target_weight=True,
        size_average: bool = True,
    ):
        super(JSDiscretLoss, self).__init__()
        self.use_target_weight = use_target_weight
        self.size_average = size_average
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def kl(self, p, q):
        """Kullback-Leibler Divergence."""

        eps = 1e-24
        kl_values = self.kl_loss((q + eps).log(), p)
        return kl_values

    def js(self, pred_hm, gt_hm):
        """Jensen-Shannon Divergence."""

        m = 0.5 * (pred_hm + gt_hm)
        js_values = 0.5 * (self.kl(pred_hm, m) + self.kl(gt_hm, m))
        return js_values

    def forward(self, pred_hm, gt_hm, target_weight=None):
        """Forward function.

        Args:
            pred_hm (torch.Tensor[N, K, H, W]): Predicted heatmaps.
            gt_hm (torch.Tensor[N, K, H, W]): Target heatmaps.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.

        Returns:
            torch.Tensor: Loss value.
        """

        if self.use_target_weight:
            assert target_weight is not None
            assert pred_hm.ndim >= target_weight.ndim

            for i in range(pred_hm.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)

            loss = self.js(pred_hm * target_weight, gt_hm * target_weight)
        else:
            loss = self.js(pred_hm, gt_hm)

        if self.size_average:
            loss /= len(gt_hm)

        return loss.sum()


@MODELS.register_module()
class KLDiscretLoss(nn.Module):
    """Discrete KL Divergence loss for SimCC with Gaussian Label Smoothing.
    Modified from `the official implementation.

    <https://github.com/leeyegy/SimCC>`_.
    Args:
        beta (float): Temperature factor of Softmax.
        label_softmax (bool): Whether to use Softmax on labels.
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
    """

    def __init__(self, beta=1.0, label_softmax=False, use_target_weight=True):
        super(KLDiscretLoss, self).__init__()
        self.beta = beta
        self.label_softmax = label_softmax
        self.use_target_weight = use_target_weight

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def criterion(self, dec_outs, labels):
        """Criterion function."""
        log_pt = self.log_softmax(dec_outs * self.beta)
        if self.label_softmax:
            labels = F.softmax(labels * self.beta, dim=1)
        loss = torch.mean(self.kl_loss(log_pt, labels), dim=1)
        return loss

    def forward(self, pred_simcc, gt_simcc, target_weight):
        """Forward function.

        Args:
            pred_simcc (Tuple[Tensor, Tensor]): Predicted SimCC vectors of
                x-axis and y-axis.
            gt_simcc (Tuple[Tensor, Tensor]): Target representations.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """
        num_joints = pred_simcc[0].size(1)
        loss = 0

        if self.use_target_weight:
            weight = target_weight.reshape(-1)
        else:
            weight = 1.

        for pred, target in zip(pred_simcc, gt_simcc):
            pred = pred.reshape(-1, pred.size(-1))
            target = target.reshape(-1, target.size(-1))

            loss += self.criterion(pred, target).mul(weight).sum()

        return loss / num_joints


@MODELS.register_module()
class InfoNCELoss(nn.Module):
    """InfoNCE loss for training a discriminative representation space with a
    contrastive manner.

    `Representation Learning with Contrastive Predictive Coding
    arXiv: <https://arxiv.org/abs/1611.05424>`_.

    Args:
        temperature (float, optional): The temperature to use in the softmax
            function. Higher temperatures lead to softer probability
            distributions. Defaults to 1.0.
        loss_weight (float, optional): The weight to apply to the loss.
            Defaults to 1.0.
    """

    def __init__(self, temperature: float = 1.0, loss_weight=1.0) -> None:
        super(InfoNCELoss, self).__init__()
        assert temperature > 0, f'the argument `temperature` must be ' \
                                f'positive, but got {temperature}'
        self.temp = temperature
        self.loss_weight = loss_weight

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Computes the InfoNCE loss.

        Args:
            features (Tensor): A tensor containing the feature
                representations of different samples.

        Returns:
            Tensor: A tensor of shape (1,) containing the InfoNCE loss.
        """
        n = features.size(0)
        features_norm = F.normalize(features, dim=1)
        logits = features_norm.mm(features_norm.t()) / self.temp
        targets = torch.arange(n, dtype=torch.long, device=features.device)
        loss = F.cross_entropy(logits, targets, reduction='sum')
        return loss * self.loss_weight


@MODELS.register_module()
class SoftMultiLabelLoss(nn.Module):
    """scenario 1: H       2-class scenario 2: H+W     2-class scenario 3:

    K*(H+W) 2-class.
    """

    def __init__(self, use_target_weight=True, mode: int = 1):
        super(SoftMultiLabelLoss, self).__init__()
        self.use_target_weight = use_target_weight
        self.mode = mode

    def criterion(self, y_pred, y_true):
        """Criterion function."""
        # N, W
        INF = -torch.tensor(np.inf, dtype=y_pred.dtype, device=y_pred.device)
        y_mask = torch.not_equal(y_pred, INF)
        y_neg = torch.where(y_mask, y_pred, INF) + torch.log(1 - y_true)
        y_pos = torch.where(y_mask, -y_pred, INF) + torch.log(y_true)
        zeros = torch.zeros_like(y_pred[..., :1])
        y_neg = torch.cat([y_neg, zeros], dim=-1)  # N, H+1
        y_pos = torch.cat([y_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_neg, dim=-1)  # N
        pos_loss = torch.logsumexp(y_pos, dim=-1)
        return neg_loss + pos_loss

    def forward(self, pred_simcc, gt_simcc, target_weight):
        """Forward function.

        Args:
            pred_simcc (Tuple[Tensor, Tensor]): Predicted SimCC vectors of
                x-axis and y-axis.
            gt_simcc (Tuple[Tensor, Tensor]): Target representations.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """
        num_joints = pred_simcc[0].size(1)
        loss = 0

        if self.use_target_weight:
            weight = target_weight.reshape(-1)
        else:
            weight = 1.

        if self.mode == 1:
            # scenario 1: H       2-class
            for pred, target in zip(pred_simcc, gt_simcc):
                pred = pred.reshape(-1, pred.size(-1))  # NK, H
                target = target.reshape(-1, target.size(-1))
                loss = loss + self.criterion(pred, target).mul(weight).mean()
        elif self.mode == 2:
            # scenario 2: H+W     2-class
            pred_x, pred_y = pred_simcc
            gt_x, gt_y = gt_simcc
            pred = torch.cat([pred_x, pred_y], dim=-1)  # N, K, H+W
            target = torch.cat([gt_x, gt_y], dim=-1)
            pred = pred.reshape(-1, pred.size(-1))  # NK, H+W
            target = target.reshape(-1, target.size(-1))
            loss = loss + self.criterion(pred, target).mul(weight).mean()
        elif self.mode == 3:
            # scenario 3: K*(H+W) 2-class
            pred_x, pred_y = pred_simcc
            gt_x, gt_y = gt_simcc
            pred = torch.cat([pred_x, pred_y], dim=-1)  # N, K, H+W
            target = torch.cat([gt_x, gt_y], dim=-1)
            N, K, _ = pred.shape
            pred = pred * weight.reshape(N, K, 1)  # N, K, H+W
            target = target * weight.reshape(N, K, 1)
            pred = pred.flatten(1)  # N, K(H+W)
            target = target.flatten(1)
            loss = loss + self.criterion(pred, target).mean()

        return loss / num_joints


@MODELS.register_module()
class HardMultilabelLoss(nn.Module):
    """scenario 1: 2 H-class scenario 2: K H-class."""

    def __init__(self, use_target_weight=True, mode=1):
        super(HardMultilabelLoss, self).__init__()
        self.use_target_weight = use_target_weight
        self.mode = mode

    # def criterion(self, pred, target):
    #     # N, K, W
    #     loss = 0.
    #     for j in range(pred.size(2)):
    #         tmp_i = 0.
    #         for k in range(pred.size(2)):
    #             if k == j:
    #                 continue
    #             up = -pred[:, :, j] + pred[:, :, k]  # N, K
    #             tmp_i = tmp_i + target[:, :, j] * torch.exp(up)
    #         tmp_i = tmp_i.sum(-1)
    #         loss = loss + torch.log(1 + tmp_i)  # N
    #     return loss
    # def criterion(self, pred, target):
    #     # N, K, W
    #     loss = 0.
    #     for j in range(pred.size(2)):
    #         up = -pred[:, :, j:j+1] + pred  # N, K, W
    #         tmp_i = target[:, :, j:j+1] * torch.exp(up)
    #         tmp_i = tmp_i.sum(-1) - target[:, :, j]
    #         loss = loss + torch.log(1 + tmp_i.sum(-1).clip_(0))  # N
    #     return loss
    # def criterion(self, pred, target):
    #     # N, K, W
    #     up = -pred.unsqueeze(-1) + pred.unsqueeze(-2)  # N, K, W, W
    #     tmp_i = target * up.exp_().sum(-1)
    #     tmp_i = tmp_i.sum(1) - target.sum(1)  # N, K, W
    #     loss = torch.log_(1 + tmp_i).sum(-1)  # N
    #     return loss
    # def criterion(self, pred, target):
    #     # N, K, W
    #     tmp_i = 0.
    #     for i in range(pred.size(-1)):
    #         t = torch.exp(-pred[:, :, i:i+1] + pred)  # N, K, W
    #         tmp_i = tmp_i + t * target[:, :, i:i+1]
    #     tmp_i = tmp_i.sum(2) - target.sum(2)  # N, K, W
    #     loss = torch.log_(1 + tmp_i).sum(-1)  # N
    #     return loss
    def criterion(self, pred, target):
        # N, K, W
        # up = torch.where(target.bool(), pred, torch.zeros_like(pred))
        up = pred * target
        up = -up.sum(-1, keepdim=True) + pred  # N, K, W
        tmp_i = torch.exp(up) - target
        loss = torch.log(1 + tmp_i.sum(-1)).sum(-1)  # N
        return loss

    def soft_criterion(self, pred, target):
        target = F.softmax(target * 10., dim=-1)
        # N, K, W
        loss = 0.
        for j in range(pred.size(2)):
            s_i = pred.mean(-1)  # N, K
            up = -pred[:, :, j] + s_i
            tmp = target[:, :, j] * torch.exp(up)
            loss = loss + torch.log(1 + tmp.sum(-1))  # N
        return loss

    def forward(self, pred_simcc, gt_simcc, target_weight):
        """Forward function.

        Args:
            pred_simcc (Tuple[Tensor, Tensor]): Predicted SimCC vectors of
                x-axis and y-axis.
            gt_simcc (Tuple[Tensor, Tensor]): Target representations.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """
        loss = 0

        if self.use_target_weight:
            weight = target_weight.unsqueeze(-1)
        else:
            weight = 1.

        if self.mode == 1:
            # scenario 1: 2 H-class
            pred = torch.stack(pred_simcc, dim=2)  # N, K, 2, H
            target = torch.stack(gt_simcc, dim=2)
            H = pred.shape[-1]
            pred = pred.reshape(-1, 2, H)  # NK, 2, H
            target = target.reshape(-1, 2, H)
            loss = self.criterion(pred, target).mul(weight).mean()
            # weight = weight.reshape(-1, pred.size(1))
            # for i in range(pred.size(1)):
            #     w = weight[:, i].reshape(-1, 1, 1)  # N
            #     loss = loss + self.criterion(pred[:, i] * w,
            #                                  target[:, i] * w).mean()
        elif self.mode == 2:
            # scenario 2: K H-class
            N, K, _ = pred_simcc[0].shape
            weight = weight.reshape(N, K, 1)
            for pred, target in zip(pred_simcc, gt_simcc):
                loss = loss + self.criterion(pred * weight,
                                             target * weight).mean()
        elif self.mode == 3:
            # scenario 3: 2K H-class
            N, K, _ = pred_simcc[0].shape
            weight = weight.reshape(N, K, 1)
            pred = torch.cat([p * weight for p in pred_simcc], dim=1)
            # N, 2K, H
            target = torch.cat([g * weight for g in gt_simcc], dim=1)
            loss = loss + self.criterion(pred, target).mean() / K

        return loss
