# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch
import torch.nn as nn

from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType


@MODELS.register_module()
class MultipleLossWrapper(nn.Module):
    """A wrapper to collect multiple loss functions together and return a list
    of losses in the same order.

    Args:
        losses (list): List of Loss Config

    Example:
        >>> losses = [
        >>>     dict(type='MSELoss', use_target_weight=True),
        >>>     dict(type='SmoothL1Loss', use_target_weight=True)
        >>> ]
        >>> self.loss = MultipleLossListWrapper(losses)
        >>>
        >>> input_list = [pred1, pred2]
        >>> target_list = [target1, target2]
        >>> losses = self.loss(input_list, target_list, target_weight)
        >>> losses[0]  # MSELoss(pred1, target1)
        >>> losses[1]  # SmoothL1Loss(pred2, target2)
    """

    def __init__(self,
                 losses: list,
                 input_mode: str = 'list',
                 loss_weights: list = [1.0, 1.0],
                 reduction: str = 'mean'):
        super().__init__()
        self.num_losses = len(losses)
        self.input_mode = input_mode
        self.loss_weights = loss_weights

        assert input_mode in [
            'list', 'single'
        ], 'input_mode should be either `list` or `single`'
        assert len(loss_weights) == self.num_losses, (
            'The length of loss_weights should be equal to the'
            ' number of losses')
        assert reduction in [
            'none', 'mean', 'sum'
        ], 'reduction should be either `none`, `mean` or `sum`'

        loss_modules = []
        for loss_cfg in losses:
            t_loss = MODELS.build(loss_cfg)
            loss_modules.append(t_loss)
        self.loss_modules = nn.ModuleList(loss_modules)

    def forward(self, input_list, target_list, keypoint_weights=None):
        """Forward function.
        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)
        Args:
            input_list (List[Tensor]): List of inputs.
            target_list (List[Tensor]): List of targets.
            keypoint_weights (Tensor[N, K, D]):
                Weights across different joint types.
        """
        assert isinstance(input_list, list), ''
        assert isinstance(target_list, list), ''
        assert len(input_list) == len(target_list), ''

        losses = []
        for i in range(self.num_losses):
            input_i = input_list[i] \
                if self.input_mode == 'list' else input_list
            target_i = target_list[i] \
                if self.input_mode == 'list' else target_list

            loss_i = self.loss_modules[i](input_i, target_i, keypoint_weights)
            loss_i = loss_i * self.loss_weights[i]
            losses.append(loss_i)

        if self.reduction == 'mean':
            losses = torch.stack(losses, dim=0).mean()
        elif self.reduction == 'sum':
            losses = torch.stack(losses, dim=0).sum()

        return losses


@MODELS.register_module()
class CombinedLoss(nn.ModuleDict):
    """A wrapper to combine multiple loss functions. These loss functions can
    have different input type (e.g. heatmaps or regression values), and can
    only be involed individually and explixitly.
    Args:
        losses (Dict[str, ConfigType]): The names and configs of loss
            functions to be wrapped
    Example::
        >>> heatmap_loss_cfg = dict(type='KeypointMSELoss')
        >>> ae_loss_cfg = dict(type='AssociativeEmbeddingLoss')
        >>> loss_module = CombinedLoss(
        ...     losses=dict(
        ...         heatmap_loss=heatmap_loss_cfg,
        ...         ae_loss=ae_loss_cfg))
        >>> loss_hm = loss_module.heatmap_loss(pred_heatmap, gt_heatmap)
        >>> loss_ae = loss_module.ae_loss(pred_tags, keypoint_indices)
    """

    def __init__(self, losses: Dict[str, ConfigType]):
        super().__init__()
        for loss_name, loss_cfg in losses.items():
            self.add_module(loss_name, MODELS.build(loss_cfg))
