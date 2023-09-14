# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple

import torch
from mmcv.cnn import Scale
from torch import Tensor, nn

from mmpose.evaluation.functional import simcc_pck_accuracy
from mmpose.registry import MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import ConfigType, OptConfigType, OptSampleList
from .rtmcc_head import RTMCCHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class MLECCHead(RTMCCHead):

    def __init__(
        self,
        dropout_rate: float = 0.0,
        loss_order: Optional[ConfigType] = None,
        adaptive_exp: str = 'none',
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        # dropout
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

        # grid
        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)
        self.field_size = (H, W)
        self.register_buffer(
            'coord_x',
            torch.linspace(0, self.input_size[0] - 1 / self.simcc_split_ratio,
                           W).view(1, 1, W))
        self.register_buffer(
            'coord_y',
            torch.linspace(0, self.input_size[1] - 1 / self.simcc_split_ratio,
                           H).view(1, 1, H))

        # sigma branch
        self.offset_transform_layer = nn.Sequential(
            nn.Linear(self.cls_x.in_features, 1), nn.Sigmoid(), Scale(10.0))

        # for order loss
        self.loss_order = MODELS.build(loss_order)
        self.x_indices = dict()
        self.y_indices = dict()

        # adaptive exp
        self.adaptive_exp = adaptive_exp
        if self.adaptive_exp != 'none':
            self.exp_mix_factor = torch.nn.Parameter(
                torch.zeros(self.out_channels).float())

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward the network.

        The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            pred_x (Tensor): 1d representation of x.
            pred_y (Tensor): 1d representation of y.
        """
        feats = feats[-1]

        feats = self.final_layer(feats)  # -> B, K, H, W

        # flatten the output heatmap
        feats = torch.flatten(feats, 2)

        feats_mlp = self.mlp(feats)  # -> B, K, hidden

        feats = self.gau(feats_mlp)

        pred_x = self.cls_x(feats)
        pred_y = self.cls_y(feats)

        if not self.training:
            return pred_x, pred_y
        else:
            sigmas = self.offset_transform_layer(feats_mlp)
            return pred_x, pred_y, sigmas

    def compute_prior(self, preds):
        """conduct dropout and softmax in 2d space (introduced in v3)"""

        preds = preds - preds.max(dim=-1, keepdims=True).values.detach()
        exp_fields = preds.exp()

        # dropout
        if hasattr(self, 'dropout'):
            exp_fields = self.dropout(exp_fields)

        # softmax
        prior_prob = exp_fields / (
            exp_fields.sum(dim=-1, keepdims=True) + 1e-8)
        return prior_prob

    def compute_posterior(self, offset, sigmas):

        dist = torch.abs(offset)  # [b, k, bins]
        dist = dist * sigmas.pow(-1)
        if self.adaptive_exp == 'none':
            scale = 0.25
        else:
            if self.adaptive_exp == 'mix':
                factor = self.exp_mix_factor.sigmoid().view(1, -1, 1)
                dist = dist * factor + dist.pow(2) * (1 - factor)
            elif self.adaptive_exp == 'exp':
                factor = self.exp_mix_factor.sigmoid().view(1, -1, 1)
                dist = dist.pow(1 + factor)
            scale = (0.25) * factor + ((2 * torch.pi)**(-0.5)) * (1 - factor)
        prob = torch.exp(-dist / 2)
        prob = prob * sigmas.pow(-1) * scale

        return prob

    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        # get gt data
        gt_xy = torch.cat(
            [d.gt_instance_labels.keypoint_labels for d in batch_data_samples],
            dim=0)
        keypoint_weights = torch.cat(
            [
                d.gt_instance_labels.keypoint_weights
                for d in batch_data_samples
            ],
            dim=0,
        )

        # compute prior
        pred_x, pred_y, sigmas = self.forward(feats)
        prior_prob_x = self.compute_prior(pred_x)
        prior_prob_y = self.compute_prior(pred_y)

        # compute posterior
        gt_x, gt_y = gt_xy.split(1, dim=-1)
        offset_x = gt_xy.narrow(2, 0, 1) - self.coord_x  # shape: [B, K, bins]
        offset_y = gt_xy.narrow(2, 1, 1) - self.coord_y
        post_prob_x = self.compute_posterior(offset_x, sigmas)
        post_prob_y = self.compute_posterior(offset_y, sigmas)

        # calculate losses
        losses = dict()
        loss = self.loss_module((prior_prob_x, prior_prob_y),
                                (post_prob_x, post_prob_y), keypoint_weights)
        losses.update(loss_kpt=loss)

        # compute order loss
        pred_x_sorted = self._sort_heatmap_by_dist(pred_x, torch.abs(offset_x))
        pred_y_sorted = self._sort_heatmap_by_dist(pred_y, torch.abs(offset_y))

        losses['loss_order'] = self.loss_order(
            (pred_x, pred_y), (pred_x_sorted, pred_y_sorted), keypoint_weights)

        # other record
        losses['sigmas'] = sigmas.mean()

        # calculate accuracy
        with torch.no_grad():
            _, avg_acc, _ = simcc_pck_accuracy(
                output=to_numpy((pred_x, pred_y)),
                target=to_numpy((
                    torch.exp(-offset_x.pow(2)),
                    torch.exp(-offset_y.pow(2)),
                )),
                simcc_split_ratio=self.simcc_split_ratio,
                mask=to_numpy(keypoint_weights) > 0,
            )

        acc_pose = torch.tensor(avg_acc, device=gt_x.device)
        losses.update(acc_pose=acc_pose)

        return losses

    @staticmethod
    @torch.no_grad()
    def _sort_heatmap_by_dist(hms, dist):
        assert hms.size() == dist.size()

        # Get the sorting indices of dist
        dist_sort_indices = dist.argsort(dim=-1)

        # Create a tensor to hold the ranks
        dist_rank = torch.zeros_like(dist_sort_indices)

        # Create a range tensor of the same size as the last dimension of dist
        range_tensor = torch.arange(
            dist.size(-1), device=dist.device).expand(*dist_sort_indices.shape)

        # Scatter range_tensor into dist_rank according to dist_sort_indices
        dist_rank.scatter_(-1, dist_sort_indices, range_tensor)

        # Sort hm in descending order
        hms_sorted_desc = hms.sort(dim=-1, descending=True)[0]

        # Use dist_rank to index into hm_sorted_desc
        hms_sorted = hms_sorted_desc.gather(-1, dist_rank)

        return hms_sorted


@MODELS.register_module()
class MLECC2DHead(MLECCHead):

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        # grid
        H, W = self.field_size
        y_coord, x_coord = torch.meshgrid(
            torch.linspace(0, self.input_size[1] - 1 / self.simcc_split_ratio,
                           H),
            torch.linspace(0, self.input_size[0] - 1 / self.simcc_split_ratio,
                           W))
        self.register_buffer(
            'coord',
            torch.stack((x_coord, y_coord), dim=-1).view(1, 1, H, W, 2))

    def compute_prior(self, pred_xy):
        """conduct dropout and softmax in 2d space (introduced in v3)"""
        pred_x, pred_y = pred_xy

        pred_fields = pred_x.unsqueeze(2) + pred_y.unsqueeze(3)
        h, w = pred_fields.shape[-2:]
        pred_fields = pred_fields.flatten(-2)

        pred_fields = pred_fields - pred_fields.max(
            dim=-1, keepdims=True).values.detach()
        exp_fields = pred_fields.exp()

        # dropout
        if hasattr(self, 'dropout'):
            exp_fields = self.dropout(exp_fields)

        # softmax
        prior_prob = exp_fields / (
            exp_fields.sum(dim=-1, keepdims=True) + 1e-8)
        prior_prob = prior_prob.reshape(*prior_prob.shape[:-1], h, w)
        return prior_prob

    def compute_posterior(self, offset, sigmas):

        dist = offset.norm(p=2, dim=-1)
        dist = dist / sigmas.unsqueeze(-1)
        if self.adaptive_exp == 'none':
            scale = 0.25
        else:
            if self.adaptive_exp == 'mix':
                factor = self.exp_mix_factor.sigmoid().view(1, -1, 1, 1)
                dist = dist * factor + dist.pow(2) * (1 - factor)
            elif self.adaptive_exp == 'exp':
                factor = self.exp_mix_factor.sigmoid().view(1, -1, 1, 1)
                dist = dist.pow(1 + factor)
            scale = (0.25) * factor + ((2 * torch.pi)**(-0.5)) * (1 - factor)

        prob = torch.exp(-dist / 2)
        prob = prob * sigmas.pow(-2).unsqueeze(-1) * scale

        return prob

    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        # get gt data
        gt_xy = torch.cat(
            [d.gt_instance_labels.keypoint_labels for d in batch_data_samples],
            dim=0)
        keypoint_weights = torch.cat(
            [
                d.gt_instance_labels.keypoint_weights
                for d in batch_data_samples
            ],
            dim=0,
        )

        # compute prior
        pred_x, pred_y, sigmas = self.forward(feats)
        prior_prob = self.compute_prior((pred_x, pred_y))

        # compute posterior
        offset = gt_xy.unsqueeze(2).unsqueeze(2) - self.coord
        post_prob = self.compute_posterior(offset, sigmas)

        # calculate losses
        losses = dict()
        loss = self.loss_module(prior_prob, post_prob, keypoint_weights)
        losses.update(loss_kpt=loss)

        # compute order loss
        offset_x = gt_xy.narrow(2, 0, 1) - self.coord_x  # shape: [B, K, bins]
        offset_y = gt_xy.narrow(2, 1, 1) - self.coord_y
        pred_x_sorted = self._sort_heatmap_by_dist(pred_x, torch.abs(offset_x))
        pred_y_sorted = self._sort_heatmap_by_dist(pred_y, torch.abs(offset_y))

        losses['loss_order'] = self.loss_order(
            (pred_x, pred_y), (pred_x_sorted, pred_y_sorted), keypoint_weights)

        # other record
        losses['sigmas'] = sigmas.mean()
        if hasattr(self, 'exp_mix_factor'):
            losses['factor'] = self.exp_mix_factor.sigmoid().mean().detach()

        # calculate accuracy
        with torch.no_grad():
            _, avg_acc, _ = simcc_pck_accuracy(
                output=to_numpy((pred_x, pred_y)),
                target=to_numpy((
                    torch.exp(-offset_x.pow(2)),
                    torch.exp(-offset_y.pow(2)),
                )),
                simcc_split_ratio=self.simcc_split_ratio,
                mask=to_numpy(keypoint_weights) > 0,
            )

        acc_pose = torch.tensor(avg_acc, device=gt_xy.device)
        losses.update(acc_pose=acc_pose)

        return losses
