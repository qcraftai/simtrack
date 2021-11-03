import numpy as np

from det3d import torchie

from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from det3d.builder import (
    build_dbsampler,
)
from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.core.utils.center_utils import (
    draw_gaussian, gaussian_radius
)
from ..registry import PIPELINES

def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]

def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


@PIPELINES.register_module
class Preprocess(object):
    def __init__(self, cfg=None, **kwargs):
        self.mode = cfg.mode
        #self.class_names = cfg.class_names
        if self.mode == "train":
            self.global_rotation_noise = cfg.global_rot_noise
            self.global_scaling_noise = cfg.global_scale_noise
            
            self.flip = cfg.flip
            if cfg.db_sampler.enable:
                self.db_sampler = build_dbsampler(cfg.db_sampler)
            else:
                self.db_sampler = None

    def __call__(self, res, info):

        res["mode"] = self.mode

        points = res["lidar"]["points"]

        if self.mode == "train":
            anno_dict = res["lidar"]["annotations"]
            gt_dict = {
                "gt_boxes": anno_dict["boxes"],
                "gt_names": np.array(anno_dict["names"]).reshape(-1),   
            }
        
            if "prev_gt_boxes" in anno_dict:
                gt_dict.update({"prev_gt_boxes": anno_dict["prev_gt_boxes"], 
                                "prev_gt_names": np.array(anno_dict['prev_gt_names']).reshape(-1),})

            selected = drop_arrays_by_name(gt_dict["gt_names"], ["DontCare", "ignore"])
            _dict_select(gt_dict, selected)
            
            if 'prev_gt_boxes' in gt_dict:
                selected = drop_arrays_by_name(gt_dict["prev_gt_names"], ["DontCare", "ignore"])
                _dict_select(gt_dict, selected)

            if self.db_sampler:
                sampled_dict = self.db_sampler.sample_all(
                    root_path = res["metadata"]["image_prefix"],
                    gt_boxes = gt_dict["gt_boxes"],
                    gt_names = gt_dict["gt_names"])

                if sampled_dict is not None:
                    gt_dict["gt_names"] = np.concatenate([gt_dict["gt_names"], sampled_dict["gt_names"]], axis=0)
                    gt_dict["gt_boxes"] = np.concatenate([gt_dict["gt_boxes"], sampled_dict["gt_boxes"]])
                    points = np.concatenate([sampled_dict["points"], points], axis=0)
                    if 'prev_gt_boxes' in gt_dict:
                        gt_dict['prev_gt_names'] = np.concatenate([gt_dict["prev_gt_names"], 
                                                np.array(['new_obj'] * len(sampled_dict["gt_names"]))], axis=0)
                        gt_dict['prev_gt_boxes'] = np.concatenate([gt_dict['prev_gt_boxes'], 
                                    np.zeros((len(sampled_dict["gt_names"]), 7))*np.nan], axis=0)
         
            # data augmentation
            if self.flip is not None:
                [gt_dict["gt_boxes"], gt_dict["prev_gt_boxes"]], points = prep.random_flip(
                    [gt_dict["gt_boxes"], gt_dict["prev_gt_boxes"]], points, flip_prob=self.flip)
           
            [gt_dict["gt_boxes"], gt_dict["prev_gt_boxes"]], points = prep.global_rotation(
                [gt_dict["gt_boxes"], gt_dict["prev_gt_boxes"]], points, rotation=self.global_rotation_noise)
            
            [gt_dict["gt_boxes"], gt_dict["prev_gt_boxes"]], points = prep.global_scaling(
                [gt_dict["gt_boxes"], gt_dict["prev_gt_boxes"]], points, *self.global_scaling_noise)
            
            # annotations
            res["lidar"]["annotations"] = gt_dict
            
        #if self.shuffle_points:
        np.random.shuffle(points) # shuffle is a little slow.
        
        res["lidar"]["points"] = points

        return res, info

@PIPELINES.register_module
class Voxelization(object):
    def __init__(self, **kwargs):
        cfg = kwargs.get("cfg", None)
        self.range = cfg.range
        self.voxel_size = cfg.voxel_size
        self.max_points_in_voxel = cfg.max_points_in_voxel
        self.max_voxel_num = cfg.max_voxel_num
        
        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num,
        )

    def __call__(self, res, info, pred_motion=None):
        voxel_size = self.voxel_generator.voxel_size
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = self.voxel_generator.grid_size

        voxels, coordinates, num_points = self.voxel_generator.generate(res["lidar"]["points"])
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
        res["lidar"]["voxels"] = dict(
            voxels=voxels,
            coordinates=coordinates,
            num_points=num_points,
            num_voxels=num_voxels,
            shape=grid_size,
            range=pc_range,
            size=voxel_size,
        )

        return res, info


def merge_multi_group_label(gt_classes, num_classes_by_task): 
    num_task = len(gt_classes)
    flag = 0 

    for i in range(num_task):
        gt_classes[i] += flag 
        flag += num_classes_by_task[i]

    return np.concatenate(gt_classes, axis=0)



@PIPELINES.register_module
class AssignTracking(object):
    def __init__(self, **kwargs):
        """Return CenterNet training labels like heatmap, height, offset"""
        assigner_cfg = kwargs["cfg"]
        self.out_size_factor = assigner_cfg.out_size_factor
        self.tasks = assigner_cfg.target_assigner.tasks
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius

    def __call__(self, res, info):
        max_objs = self._max_objs
        class_names_to_id = {}
        for ti in range(len(self.tasks)):
            t = self.tasks[ti]
            for ni in range(t['num_class']): # task id, cat_id(in task)
                class_names_to_id[self.tasks[ti]['class_names'][ni]] = [ti, ni]
        
        # Calculate output featuremap size
        grid_size = res["lidar"]["voxels"]["shape"]
        pc_range = res["lidar"]["voxels"]["range"]
        voxel_size = res["lidar"]["voxels"]["size"]

        feature_map_size = grid_size[:2] // self.out_size_factor
        
        tracking_example = {}

        gt_dict = res["lidar"]["annotations"]

        tracking_hms, tracking_annos, tracking_inds, tracking_masks, tracking_cats = [], [], [], [], []

        for idx, task in enumerate(self.tasks):
            prev_hm = np.zeros((task['num_class'], feature_map_size[1], feature_map_size[0]), dtype=np.float32)
            anno_box = np.zeros((max_objs, 10), dtype=np.float32) 
            ind = np.zeros((max_objs), dtype=np.int64)
            mask = np.zeros((max_objs), dtype=np.uint8)
            cat = np.zeros((max_objs), dtype=np.int64)
            
            tracking_hms.append(prev_hm)
            tracking_annos.append(anno_box)
            tracking_inds.append(ind)
            tracking_masks.append(mask)
            tracking_cats.append(cat)

        task_nums = np.zeros(len(self.tasks), dtype=np.int64)
        for k in range(len(gt_dict['prev_gt_names'])):
            obj_name = gt_dict['gt_names'][k]
            if obj_name not in class_names_to_id:
                continue
            if gt_dict['prev_gt_names'][k] == 'new_obj':
                x, y = gt_dict['gt_boxes'][k][0], gt_dict['gt_boxes'][k][1]
                size_x, size_y = gt_dict['gt_boxes'][k][3], gt_dict['gt_boxes'][k][4]
            else:
                x, y = gt_dict['prev_gt_boxes'][k][0], gt_dict['prev_gt_boxes'][k][1]
                size_x, size_y = gt_dict['prev_gt_boxes'][k][3], gt_dict['prev_gt_boxes'][k][4]
            
            size_x, size_y = size_x / voxel_size[0] / self.out_size_factor, size_y / voxel_size[1] / self.out_size_factor 

            if size_x > 0 and size_y > 0 and obj_name != 'disappear':
                task_id = class_names_to_id[obj_name][0]
                cls_id = class_names_to_id[obj_name][1]
                radius = gaussian_radius((size_y, size_x), min_overlap=self.gaussian_overlap)
                radius = max(self._min_radius, int(radius))

                # be really careful for the coordinate system of your box annotation. 
                coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
                                (y - pc_range[1]) / voxel_size[1] / self.out_size_factor

                ct = np.array([coor_x, coor_y], dtype=np.float32)  
                ct_int = ct.astype(np.int32)
                
                # throw out not in range objects to avoid out of array area when creating the heatmap
                if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                    continue 
                
                draw_gaussian(tracking_hms[task_id][cls_id], ct, radius, 1.0) 
                
                new_idx = task_nums[task_id]
                x, y = ct_int[0], ct_int[1]  # int voxel coordinate  # first appear location 
              
                tracking_cats[task_id][new_idx] = cls_id
                    
                tracking_inds[task_id][new_idx] = y * feature_map_size[0] + x
                tracking_masks[task_id][new_idx] = 1

                vx, vy = gt_dict['gt_boxes'][k][6:8]
                rot = gt_dict['gt_boxes'][k][8]

                new_x, new_y = gt_dict['gt_boxes'][k][0], gt_dict['gt_boxes'][k][1]

                new_coor_x, new_coor_y = (new_x - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
                                (new_y - pc_range[1]) / voxel_size[1] / self.out_size_factor

                ct = np.array([new_coor_x, new_coor_y], dtype=np.float32)  
               
                tracking_annos[task_id][new_idx] = np.concatenate(
                        (ct - (x, y), gt_dict['gt_boxes'][k][2], np.log(gt_dict['gt_boxes'][k][3:6]),
                        np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                
                task_nums[task_id] = task_nums[task_id] + 1
        
        tracking_example.update({'hm': tracking_hms, 'anno_box': tracking_annos, 'ind': tracking_inds, 'mask': tracking_masks, 'cat': tracking_cats}) 
        res["lidar"]["targets"] = tracking_example                    
        return res, info

@PIPELINES.register_module
class AssignLabel(object):
    def __init__(self, **kwargs):
        """Return CenterNet training labels like heatmap, height, offset"""
        assigner_cfg = kwargs["cfg"]
        self.out_size_factor = assigner_cfg.out_size_factor
        self.tasks = assigner_cfg.target_assigner.tasks
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius

    def __call__(self, res, info):
        max_objs = self._max_objs
        class_names_to_id = {}
        for ti in range(len(self.tasks)):
            t = self.tasks[ti]
            for ni in range(t['num_class']): # task id, cat_id(in task)
                class_names_to_id[self.tasks[ti]['class_names'][ni]] = [ti, ni]
        
        # Calculate output featuremap size
        grid_size = res["lidar"]["voxels"]["shape"]
        pc_range = res["lidar"]["voxels"]["range"]
        voxel_size = res["lidar"]["voxels"]["size"]

        feature_map_size = grid_size[:2] // self.out_size_factor
        
        det_example = {}

        gt_dict = res["lidar"]["annotations"]

        curr_hms, curr_annos, curr_inds, curr_masks, curr_cats = [], [], [], [], []

        for idx, task in enumerate(self.tasks):
            curr_hm = np.zeros((task['num_class'], feature_map_size[1], feature_map_size[0]), dtype=np.float32)
            anno_box = np.zeros((max_objs, 10), dtype=np.float32) 
            ind = np.zeros((max_objs), dtype=np.int64)
            mask = np.zeros((max_objs), dtype=np.uint8)
            cat = np.zeros((max_objs), dtype=np.int64)
            
            curr_hms.append(curr_hm)
            curr_annos.append(anno_box)
            curr_inds.append(ind)
            curr_masks.append(mask)
            curr_cats.append(cat)

        task_nums = np.zeros(len(self.tasks), dtype=np.int64)
        for k in range(len(gt_dict['gt_names'])):
            obj_name = gt_dict['gt_names'][k]
            if obj_name not in class_names_to_id:
                continue
            if obj_name == 'disappear':
                continue

            x, y = gt_dict['gt_boxes'][k][0], gt_dict['gt_boxes'][k][1]
            size_x, size_y = gt_dict['gt_boxes'][k][3], gt_dict['gt_boxes'][k][4]
           
            size_x, size_y = size_x / voxel_size[0] / self.out_size_factor, size_y / voxel_size[1] / self.out_size_factor 

            if size_x > 0 and size_y > 0:
                task_id = class_names_to_id[obj_name][0]
                cls_id = class_names_to_id[obj_name][1]
                radius = gaussian_radius((size_y, size_x), min_overlap=self.gaussian_overlap)
                radius = max(self._min_radius, int(radius))

                # be really careful for the coordinate system of your box annotation. 
                coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
                                (y - pc_range[1]) / voxel_size[1] / self.out_size_factor

                ct = np.array([coor_x, coor_y], dtype=np.float32)  
                ct_int = ct.astype(np.int32)
                
                # throw out not in range objects to avoid out of array area when creating the heatmap
                if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                    continue 
                
                draw_gaussian(curr_hms[task_id][cls_id], ct, radius, 1.0) 
                
                new_idx = task_nums[task_id]
                x, y = ct_int[0], ct_int[1]  # int voxel coordinate  # first appear location 
              
                curr_cats[task_id][new_idx] = cls_id
                    
                curr_inds[task_id][new_idx] = y * feature_map_size[0] + x
                curr_masks[task_id][new_idx] = 1

                vx, vy = gt_dict['gt_boxes'][k][6:8]
                rot = gt_dict['gt_boxes'][k][8]

                curr_annos[task_id][new_idx] = np.concatenate(
                        (ct - (x, y), gt_dict['gt_boxes'][k][2], np.log(gt_dict['gt_boxes'][k][3:6]),
                        np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                
                task_nums[task_id] = task_nums[task_id] + 1
        
        det_example.update({'curr_hm': curr_hms, 'curr_anno_box': curr_annos, 'curr_ind': curr_inds, 'curr_mask': curr_masks, 'curr_cat': curr_cats}) 
        res["lidar"]["curr_targets"] = det_example                    
        return res, info





