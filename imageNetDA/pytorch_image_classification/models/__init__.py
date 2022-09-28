import importlib

import torch
import torch.nn as nn
import torch.distributed as dist
import yacs.config


def create_model(config: yacs.config.CfgNode, teacher_dict) -> nn.Module:
    module = importlib.import_module(
        'pytorch_image_classification.models'
        f'.{config.model.type}.{config.model.name}')
    model = getattr(module, 'Network')(config, teacher_dict)
    device = torch.device(config.device)
    model.to(device)
    return model


def apply_data_parallel_wrapper(config: yacs.config.CfgNode,
                                model: nn.Module) -> nn.Module:
    local_rank = config.train.dist.local_rank
    if dist.is_available() and dist.is_initialized():
        if config.train.dist.use_sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    else:
        if config.device == 'cuda':
            model = nn.DataParallel(model)
    return model
