# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import math
from typing import Iterator, List, Optional, Sized, Union

import torch
from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.logging import MessageHub
from torch.utils.data import Sampler

from mmpose.datasets import CombinedDataset
from mmpose.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class MultiSourceSampler(Sampler):
    """Multi-Source Sampler. According to the sampling ratio, sample data from
    different datasets to form batches.

    Args:
        dataset (Sized): The dataset
        batch_size (int): Size of mini-batch
        source_ratio (list[int | float]): The sampling ratio of different
            source datasets in a mini-batch
        shuffle (bool): Whether shuffle the dataset or not. Defaults to
            ``True``
        round_up (bool): Whether to add extra samples to make the number of
            samples evenly divisible by the world size. Defaults to True.
        seed (int, optional): Random seed. If ``None``, set a random seed.
            Defaults to ``None``
    """

    def __init__(self,
                 dataset: Sized,
                 batch_size: int,
                 source_ratio: List[Union[int, float]] = None,
                 shuffle: bool = True,
                 round_up: bool = True,
                 seed: Optional[int] = None) -> None:

        assert isinstance(dataset, CombinedDataset),\
            f'The dataset must be CombinedDataset, but get {dataset}'
        assert isinstance(batch_size, int) and batch_size > 0, \
            'batch_size must be a positive integer value, ' \
            f'but got batch_size={batch_size}'
        assert isinstance(source_ratio, list), \
            f'source_ratio must be a list, but got source_ratio={source_ratio}'
        assert len(source_ratio) == len(dataset._lens), \
            'The length of source_ratio must be equal to ' \
            f'the number of datasets, but got source_ratio={source_ratio}'

        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.cumulative_sizes = [0] + list(itertools.accumulate(dataset._lens))
        self.batch_size = batch_size

        if source_ratio is not None:
            self.source_ratio = source_ratio
        else:
            self.source_ratio = [len(d) for d in self.dataset]

        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / world_size))
        self.num_per_source = [
            int(batch_size * sr / sum(self.source_ratio))
            for sr in self.source_ratio
        ]
        self.num_per_source[0] = batch_size - sum(self.num_per_source[1:])

        assert sum(self.num_per_source) == batch_size, \
            'The sum of num_per_source must be equal to ' \
            f'batch_size, but get {self.num_per_source}'

        self.seed = sync_random_seed() if seed is None else seed
        self.shuffle = shuffle
        self.round_up = round_up
        self.source2inds = {
            source: self._indices_of_rank(len(ds))
            for source, ds in enumerate(dataset.datasets)
        }

    def _infinite_indices(self, sample_size: int) -> Iterator[int]:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(sample_size, generator=g).tolist()
            else:
                yield from torch.arange(sample_size).tolist()

    def _indices_of_rank(self, sample_size: int) -> Iterator[int]:
        """Slice the infinite indices by rank."""
        yield from itertools.islice(
            self._infinite_indices(sample_size), self.rank, None,
            self.world_size)

    def __iter__(self) -> Iterator[int]:
        batch_buffer = []
        num_iters = self.num_samples // self.batch_size
        if self.round_up and self.num_samples > num_iters * self.batch_size:
            num_iters += 1
        for i in range(num_iters):
            for source, num in enumerate(self.num_per_source):
                batch_buffer_per_source = []
                for idx in self.source2inds[source]:
                    idx += self.cumulative_sizes[source]
                    batch_buffer_per_source.append(idx)
                    if len(batch_buffer_per_source) == num:
                        batch_buffer += batch_buffer_per_source
                        break
        return iter(batch_buffer)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Compatible in `epoch-based runner."""
        pass


@DATA_SAMPLERS.register_module()
class DynamicMultiSourceSampler(MultiSourceSampler):

    def __init__(self,
                 dataset: Sized,
                 batch_size: int,
                 source_ratio: List[Union[int, float]] = None,
                 mapping_metric_dataset: dict = None,
                 resample_interval: int = 1,
                 shuffle: bool = True,
                 round_up: bool = True,
                 seed: Optional[int] = None) -> None:
        super().__init__(dataset, batch_size, source_ratio, shuffle, round_up,
                         seed)
        self.resample_interval = resample_interval
        self.mapping_metric_dataset = mapping_metric_dataset
        self.ori_source_ratio = source_ratio

    def resample(self, mh):
        # examples:
        # mapping_metric_dastaset = {
        #     'coco-wholebody/body': [0, 1, 6],
        #     'coco-wholebody/foot': [2, 6],
        #     'coco-wholebody/face': [3, 4, 6],
        #     'coco-wholebody/lhand': [5, 6],
        #     'coco-wholebody/rhand': [5, 6]
        # }
        metrics = {m: mh.get_info(m) for m in self.mapping_metric_dataset}

        lens_ = {}
        sums = {}
        for k, v in self.mapping_metric_dataset.items():
            if self.ori_source_ratio is not None:
                sums[k] = sum([self.ori_source_ratio[i] for i in v])
            else:
                sums[k] = sum([len(self.dataset[i]) for i in v])
            lens_[k] = sums[k] / metrics[k]

        new_source_ratio = [0.] * len(self.dataset)
        for k, v in self.mapping_metric_dataset.items():
            for i in v:
                new_source_ratio[i] += lens_[k] / sums[k] * len(
                    self.dataset[i])

        self.source_ratio = new_source_ratio

        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.world_size))
        self.num_per_source = [
            int(self.batch_size * sr / sum(self.source_ratio))
            for sr in self.source_ratio
        ]
        self.num_per_source[0] = self.batch_size - sum(self.num_per_source[1:])

        self.source2inds = {
            source: self._indices_of_rank(len(ds))
            for source, ds in enumerate(self.dataset.datasets)
        }

    def __iter__(self) -> Iterator[int]:
        batch_buffer = []
        num_iters = self.num_samples // self.batch_size
        if self.round_up and self.num_samples > num_iters * self.batch_size:
            num_iters += 1

        mh = MessageHub.get_current_instance()
        cur_epoch = mh.get_info('epoch')
        if (cur_epoch + 1) % self.resample_interval == 0:
            self.resample(mh)

        for i in range(num_iters):
            for source, num in enumerate(self.num_per_source):
                batch_buffer_per_source = []
                for idx in self.source2inds[source]:
                    idx += self.cumulative_sizes[source]
                    batch_buffer_per_source.append(idx)
                    if len(batch_buffer_per_source) == num:
                        batch_buffer += batch_buffer_per_source
                        break
        return iter(batch_buffer)
