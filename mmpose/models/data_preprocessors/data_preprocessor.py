# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Optional, Sequence, Union

# import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import ImgDataPreprocessor
from mmengine.model.utils import stack_batch
from mmengine.utils import is_seq_of

from mmpose.registry import MODELS

Number = Union[int, float]


@MODELS.register_module()
class PoseDataPreprocessor(ImgDataPreprocessor):
    """Image pre-processor for pose estimation tasks."""

    def __init__(self,
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 non_blocking: Optional[bool] = False,
                 batch_augments: Optional[List[dict]] = None):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking)

        if batch_augments is not None:
            self.batch_augments = nn.ModuleList(
                [MODELS.build(aug) for aug in batch_augments])
        else:
            self.batch_augments = None

    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        """Performs normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataset. If the collate
                function of DataLoader is :obj:`pseudo_collate`, data will be a
                list of dict. If collate function is :obj:`default_collate`,
                data will be a tuple with batch input tensor and list of data
                samples.
            training (bool): Whether to enable training time augmentation. If
                subclasses override this method, they can perform different
                preprocessing strategies for training and testing based on the
                value of ``training``.

        Returns:
            dict or list: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        _batch_inputs, data_samples = data['inputs'], data['data_samples']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            if self.pad_size_divisor == 1:
                batch_inputs = [
                    _batch_input.float() for _batch_input in _batch_inputs
                ]
                # Pad and stack Tensor.
                batch_inputs = stack_batch(batch_inputs, self.pad_size_divisor,
                                           self.pad_value)
                # channel transform
                if self._channel_conversion:
                    batch_inputs = batch_inputs[:, [2, 1, 0], ...]

                if training and self.batch_augments is not None:
                    for each_batch_aug in self.batch_augments:
                        batch_inputs, data_samples = each_batch_aug(
                            batch_inputs, data_samples)

                # Normalization.
                if self._enable_normalize:
                    if self.mean.shape[0] == 3:
                        assert batch_inputs.dim(
                        ) == 4 and batch_inputs.shape[1] == 3, (
                            'If the mean has 3 values, the input tensor '
                            'should in shape of (3, H, W), but got the tensor '
                            f'with shape {batch_inputs.shape}')
                    batch_inputs = (batch_inputs - self.mean) / self.std
            else:
                batch_inputs = []
                for _batch_input in _batch_inputs:
                    # channel transform
                    if self._channel_conversion:
                        _batch_input = _batch_input[[2, 1, 0], ...]
                    # Convert to float after channel conversion to ensure
                    # efficiency
                    _batch_input = _batch_input.float()
                    # Normalization.
                    if self._enable_normalize:
                        _batch_input = (_batch_input - self.mean) / self.std
                    batch_inputs.append(_batch_input)
                # Pad and stack Tensor.
                batch_inputs = stack_batch(batch_inputs, self.pad_size_divisor,
                                           self.pad_value)
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            if self._channel_conversion:
                _batch_inputs = _batch_inputs[:, [2, 1, 0], ...]
            # Convert to float after channel conversion to ensure
            # efficiency
            _batch_inputs = _batch_inputs.float()

            if training and self.batch_augments is not None:
                for each_batch_aug in self.batch_augments:
                    _batch_inputs, data_samples = each_batch_aug(
                        _batch_inputs, data_samples)

            if self._enable_normalize:
                _batch_inputs = (_batch_inputs - self.mean) / self.std
            h, w = _batch_inputs.shape[2:]
            target_h = math.ceil(
                h / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(
                w / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            batch_inputs = F.pad(_batch_inputs, (0, pad_w, 0, pad_h),
                                 'constant', self.pad_value)
        else:
            raise TypeError('Output of `cast_data` should be a dict of '
                            'list/tuple with inputs and data_samples, '
                            f'but got {type(_batch_inputs)} {data.keys()}')
        data['inputs'] = batch_inputs
        data.setdefault('data_samples', None)
        return data
