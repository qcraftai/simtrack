import collections
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from .data_container import DataContainer


def collate(batch_list, samples_per_gpu=1):
    example_merged = collections.defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    batch_size = len(batch_list)
    ret = {}
    for key, elems in example_merged.items():
        if key in ["voxels", "num_points", "num_gt", "voxel_labels", "num_voxels", "dense_voxels",
                    "curr_voxels", "curr_num_points", "curr_num_voxels"]:
            ret[key] = torch.tensor(np.concatenate(elems, axis=0))
        elif key in [
            "gt_boxes",
        ]:
            task_max_gts = []
            for task_id in range(len(elems[0])):
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, len(elems[k][task_id]))
                task_max_gts.append(max_gt)
            res = []
            for idx, max_gt in enumerate(task_max_gts):
                batch_task_gt_boxes3d = np.zeros((batch_size, max_gt, 7))
                for i in range(batch_size):
                    batch_task_gt_boxes3d[i, : len(elems[i][idx]), :] = elems[i][idx]
                res.append(batch_task_gt_boxes3d)
            ret[key] = res
        elif key == "metadata":
            ret[key] = elems
        elif key == "calib":
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = torch.tensor(np.stack(v1, axis=0))
        elif key in ["coordinates", "points", "curr_coordinates"]:
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                )
                coors.append(coor_pad)
            ret[key] = torch.tensor(np.concatenate(coors, axis=0))
        elif key in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels", "hm", "prev_hm", "curr_hm", "anno_box",
                    "ind", "mask", "cat", "curr_anno_box", "curr_ind", "curr_mask", "curr_cat"]:
            ret[key] = defaultdict(list)
            res = []
            for elem in elems:
                for idx, ele in enumerate(elem):
                    ret[key][str(idx)].append(torch.tensor(ele))
            for kk, vv in ret[key].items():
                res.append(torch.stack(vv))
            ret[key] = res
        elif key in ["ogm", "motion", "car_from_global", "ref_from_car",'vehicle_to_global']:
            ret[key] = torch.tensor(np.stack(elems, axis=0)).float()
        else:
            ret[key] = np.stack(elems, axis=0)

    return ret
