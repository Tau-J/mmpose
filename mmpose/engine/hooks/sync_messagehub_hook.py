# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dist import broadcast_object_list, get_rank
from mmengine.hooks import Hook
from mmengine.logging import MessageHub
from mmengine.registry import HOOKS


@HOOKS.register_module()
class SyncMessageHubHook(Hook):
    priority = 'LOWEST'

    def __init__(self, keys: list):
        self.keys = keys

    def after_val_epoch(self, runner, metrics) -> None:
        mh = MessageHub.get_current_instance()
        values = []
        for key in self.keys:
            if get_rank() == 0:
                values.append(mh.get_info(key))
            else:
                values.append(None)

        broadcast_object_list(values, 0)

        for key, value in zip(self.keys, values):
            if get_rank() != 0:
                mh.update_info(key, value)
            print(f'rank: {get_rank()}, key: {key}, value: {value}')
