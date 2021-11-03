import logging
import pickle
from det3d.core.sampler.sample_ops import (DataBasePreprocessor, DBFilterByDifficulty, DBFilterByMinNumPoint)
from det3d.core.sampler.sample_ops import DataBaseSamplerV2
from det3d.solver import learning_schedules_fastai as lsf

def build_db_preprocess(db_prep_config, logger=None):
    logger = logging.getLogger("build_db_preprocess")
    cfg = db_prep_config
    if "filter_by_difficulty" in cfg:
        v = cfg["filter_by_difficulty"]
        return DBFilterByDifficulty(v, logger=logger)
    elif "filter_by_min_num_points" in cfg:
        v = cfg["filter_by_min_num_points"]
        return DBFilterByMinNumPoint(v, logger=logger)
    else:
        raise ValueError("unknown database prep type")


def _create_learning_rate_scheduler(optimizer, learning_rate_config, total_step):
    """Create optimizer learning rate scheduler based on config.

    Args:
        learning_rate_config: A LearningRate proto message.

    Returns:
        A learning rate.

    Raises:
        ValueError: when using an unsupported input data type.
    """
    lr_scheduler = None
    learning_rate_type = learning_rate_config.type
    config = learning_rate_config

    if learning_rate_type == "multi_phase":
        lr_phases = []
        mom_phases = []
        for phase_cfg in config.phases:
            lr_phases.append((phase_cfg.start, phase_cfg.lambda_func))
            mom_phases.append((phase_cfg.start, phase_cfg.momentum_lambda_func))
        lr_scheduler = lsf.LRSchedulerStep(optimizer, total_step, lr_phases, mom_phases)
    elif learning_rate_type == "one_cycle":
        lr_scheduler = lsf.OneCycle(
            optimizer,
            total_step,
            config.lr_max,
            config.moms,
            config.div_factor,
            config.pct_start,
        )
    elif learning_rate_type == "exponential_decay":
        lr_scheduler = lsf.ExponentialDecay(
            optimizer,
            total_step,
            config.initial_learning_rate,
            config.decay_length,
            config.decay_factor,
            config.staircase,
        )
    elif learning_rate_type == "manual_stepping":
        lr_scheduler = lsf.ManualStepping(
            optimizer, total_step, config.boundaries, config.rates
        )
    elif lr_scheduler is None:
        raise ValueError("Learning_rate %s not supported." % learning_rate_type)

    return lr_scheduler



def build_dbsampler(cfg, logger=None):
    logger = logging.getLogger("build_dbsampler")
    prepors = [build_db_preprocess(c, logger=logger) for c in cfg.db_prep_steps]
    db_prepor = DataBasePreprocessor(prepors)
    rate = cfg.rate
    gt_drop_percentage = cfg.gt_drop_percentage
    gt_drop_max_keep_points = cfg.gt_drop_max_keep_points
    groups = cfg.sample_groups
    point_dim = cfg.point_dim

    info_path = cfg.db_info_path
    with open(info_path, "rb") as f:
        db_infos = pickle.load(f)
    sampler = DataBaseSamplerV2(
        db_infos, groups, db_prepor, rate, gt_drop_percentage, gt_drop_max_keep_points, 
        point_dim, logger=logger
    )

    return sampler
