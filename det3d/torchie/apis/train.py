from __future__ import division

import re
from collections import OrderedDict, defaultdict
from functools import partial

import apex
import numpy as np
import torch
from det3d.builder import _create_learning_rate_scheduler

from det3d.core import DistOptimizerHook
from det3d.datasets import DATASETS, build_dataloader
from det3d.solver.fastai_optim import OptimWrapper
from det3d.torchie.trainer import DistSamplerSeedHook, Trainer, obj_from_dict
from det3d.utils.print_utils import metric_to_str
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .env import get_root_logger



def flatten_model(m):
    return sum(map(flatten_model, m.children()), []) if len(list(m.children())) else [m]


def get_layer_groups(m):
    return [nn.Sequential(*flatten_model(m))]


def build_one_cycle_optimizer(model, optimizer_config):
    if optimizer_config.fixed_wd:
        optimizer_func = partial(
            torch.optim.Adam, betas=(0.9, 0.99), amsgrad=optimizer_config.amsgrad
        )
    else:
        optimizer_func = partial(torch.optim.Adam, amsgrad=optimizer_cfg.amsgrad)

    optimizer = OptimWrapper.create(
        optimizer_func,
        3e-3,
        get_layer_groups(model),
        wd=optimizer_config.wd,
        true_wd=optimizer_config.fixed_wd,
        bn_wd=True,
    )

    return optimizer


def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.
    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with 3 accepted fileds
                  (bias_lr_mult, bias_decay_mult, norm_decay_mult).
                  `bias_lr_mult` and `bias_decay_mult` will be multiplied to
                  the lr and weight decay respectively for all bias parameters
                  (except for the normalization layers), and
                  `norm_decay_mult` will be multiplied to the weight decay
                  for all weight and bias parameters of normalization layers.
    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    if hasattr(model, "module"):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop("paramwise_options", None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        return obj_from_dict(
            optimizer_cfg, torch.optim, dict(params=model.parameters())
        )
    else:
        assert isinstance(paramwise_options, dict)
        # get base lr and weight decay
        base_lr = optimizer_cfg["lr"]
        base_wd = optimizer_cfg.get("weight_decay", None)
        # weight_decay must be explicitly specified if mult is specified
        if (
            "bias_decay_mult" in paramwise_options
            or "norm_decay_mult" in paramwise_options
        ):
            assert base_wd is not None
        # get param-wise options
        bias_lr_mult = paramwise_options.get("bias_lr_mult", 1.0)
        bias_decay_mult = paramwise_options.get("bias_decay_mult", 1.0)
        norm_decay_mult = paramwise_options.get("norm_decay_mult", 1.0)
        # set param-wise lr and weight decay
        params = []
        for name, param in model.named_parameters():
            param_group = {"params": [param]}
            if not param.requires_grad:
                # FP16 training needs to copy gradient/weight between master
                # weight copy and model weight, it is convenient to keep all
                # parameters here to align with model.parameters()
                params.append(param_group)
                continue

            # for norm layers, overwrite the weight decay of weight and bias
            # TODO: obtain the norm layer prefixes dynamically
            if re.search(r"(bn|gn)(\d+)?.(weight|bias)", name):
                if base_wd is not None:
                    param_group["weight_decay"] = base_wd * norm_decay_mult
            # for other layers, overwrite both lr and weight decay of bias
            elif name.endswith(".bias"):
                param_group["lr"] = base_lr * bias_lr_mult
                if base_wd is not None:
                    param_group["weight_decay"] = base_wd * bias_decay_mult
            # otherwise use the global settings

            params.append(param_group)

        optimizer_cls = getattr(torch.optim, optimizer_cfg.pop("type"))
        return optimizer_cls(params, **optimizer_cfg)


def train_detector(model, dataset, cfg, distributed=False, validate=False, logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds, cfg.data.samples_per_gpu, cfg.data.workers_per_gpu, dist=distributed
        )
        for ds in dataset
    ]

    total_steps = cfg.total_epochs * len(data_loaders[0])
    # print(f"total_steps: {total_steps}")

    if cfg.lr_config.type == "one_cycle":
        # build trainer
        optimizer = build_one_cycle_optimizer(model, cfg.optimizer)
        lr_scheduler = _create_learning_rate_scheduler(
            optimizer, cfg.lr_config, total_steps
        )
        cfg.lr_config = None
    else:
        optimizer = build_optimizer(model, cfg.optimizer)
        lr_scheduler = None
        #lr_scheduler = _create_learning_rate_scheduler(
        #    optimizer, cfg.lr_config, total_steps
        #)

    # put model on gpus
    if distributed:
        if cfg.use_syncbn:
            model = apex.parallel.convert_syncbn_model(model)

        #model, optimizer = apex.amp.initialize(model, optimizer,
        #                              opt_level=args.opt_level,
        #                              keep_batchnorm_fp32=True,
        #                              loss_scale=args.loss_scale
        #                              )
        #model = apex.parallel.DistributedDataParallel(model.cuda())
        
        model = DistributedDataParallel(
            model.cuda(cfg.local_rank),
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            # broadcast_buffers=False,
            find_unused_parameters=True,
        )
        
    else:
        model = model.cuda()

    logger.info(f"model structure: {model}")

    trainer = Trainer(
        model, optimizer, lr_scheduler, cfg.work_dir, cfg.log_level
    )

    if distributed:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    trainer.register_training_hooks(
        cfg.lr_config, optimizer_config, cfg.checkpoint_config, cfg.log_config
    )

    if distributed:
        trainer.register_hook(DistSamplerSeedHook())


    if cfg.resume_from:
        trainer.resume(cfg.resume_from)
    elif cfg.load_from:
        trainer.load_checkpoint(cfg.load_from)

    trainer.run(data_loaders, cfg.workflow, cfg.total_epochs, local_rank=cfg.local_rank)
