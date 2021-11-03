import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor

norm_cfg = None

tasks = [
    dict(num_class=1, class_names=["car"], stride=1),
    dict(num_class=2, class_names=["truck", "construction_vehicle"], stride=1),
    dict(num_class=2, class_names=["bus", "trailer"], stride=1),
    dict(num_class=1, class_names=["barrier"], stride=1),
    dict(num_class=2, class_names=["motorcycle", "bicycle"], stride=1),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"], stride=1),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))



_voxel_size = (0.2, 0.2, 8)
_pc_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
# model settings
model = dict(
    type="PointPillars",
    pretrained=None,
    reader=dict(
        type="PillarFeatureNet",
        num_input_features=5,
        num_filters=[64, 64],
        with_distance=False,
        voxel_size=_voxel_size,
        pc_range=_pc_range,
        norm_cfg=norm_cfg,
    ),
    backbone=dict(type="PointPillarsScatter", num_input_features=64, norm_cfg=norm_cfg, ds_factor=1),
    neck=dict(
        type="RPN",
        layer_nums=[3, 5, 5],
        ds_layer_strides=[2, 2, 2],
        ds_num_filters=[64, 128, 256],
        us_layer_strides=[0.5, 1, 2], # #[1, 2, 4], #, 
        us_num_filters=[128, 128, 128],
        num_input_features=64,
        norm_cfg=norm_cfg,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHeadV2", 
        in_channels=sum([128, 128, 128]),  # this is linked to 'neck' us_num_filters
        tasks=tasks,
        weight=0.25,
        code_weights=[4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2), 'vel': (2, 2)},
    ),
)

target_assigner = dict(tasks=tasks)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
)

train_cfg = dict(assigner=assigner)

test_cfg = dict(
    nms=dict(
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.2,
    ),
    score_threshold=0.1,
    pc_range=[-51.2, -51.2],
    out_size_factor=get_downsample_factor(model),
    voxel_size=[0.2, 0.2],
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
)

# dataset settings
dataset_type = "NuScenesDataset"
n_sweeps = 10
data_root = "data/v1.0-trainval"

db_sampler = dict(
    type="GT-AUG",
    enable=True, 
    db_info_path="data/v1.0-trainval/dbinfos_train_1sweeps.pkl",
    sample_groups=[
        dict(car=2),
        dict(truck=3),
        dict(construction_vehicle=7),
        dict(bus=4),
        dict(trailer=6),
        #dict(barrier=6),
        dict(motorcycle=2),
        dict(bicycle=6),
        dict(pedestrian=2),
        #dict(traffic_cone=2),
    ],
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                car=5,
                truck=5,
                bus=5,
                trailer=5,
                construction_vehicle=5,
                traffic_cone=5,
                barrier=5,
                motorcycle=5,
                bicycle=5,
                pedestrian=5,
            )
        ),
        dict(filter_by_difficulty=[-1],),
    ],
    rate=1.0,
    gt_drop_percentage=0.5,
    gt_drop_max_keep_points=5,
    point_dim=5,   
)

train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.3925, 0.3925],
    global_scale_noise=[0.95, 1.05],
    global_trans_noise=[0.2, 0.2, 0.2],
    remove_points_after_sample=False,
    remove_unknown_examples=False,
    min_points_in_gt=0, 
    flip=[0.5, 0.5],
    db_sampler=db_sampler,
    class_names=class_names,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=True, 
    remove_environment=False,
    remove_unknown_examples=False,
    class_names=class_names,
)

voxel_generator = dict(
    range=_pc_range,
    voxel_size=_voxel_size,
    max_points_in_voxel=20,  
    max_voxel_num=30000, 
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type, nsweeps=n_sweeps),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignTracking", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type, nsweeps=n_sweeps),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="Reformat"),
]

train_anno = "data/v1.0-trainval/infos_train_10sweeps_tracking.pkl"
val_anno = "data/v1.0-trainval/infos_val_10sweeps_tracking.pkl"
test_anno = None

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=10,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        n_sweeps=n_sweeps,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        ann_file=val_anno,
        test_mode=True,
        n_sweeps=n_sweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        n_sweeps=n_sweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)

# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)

"""training hooks """
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy in training hooks
lr_config = dict(
    type="one_cycle", lr_max=0.004, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

use_syncbn = True 

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
# yapf:enable
# runtime settings
total_epochs = 20
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = "./experiments/pointpillars"
load_from = None 
resume_from = None
workflow = [("train", 1), ("val", 1)]
