
import sys
import time
from collections import OrderedDict
from functools import reduce

import numba
import numpy as np

from det3d.core.bbox import box_np_ops
import copy

# corrected
def random_flip(gt_boxes_list, points, flip_prob=[0.5, 0.5], ):
    assert 0 <= flip_prob[0] <=1
    assert 0 <= flip_prob[1] <=1

    # x flip
    if flip_prob[0] > 0:
        enable = np.random.choice(
            [False, True], replace=False, p=[1 - flip_prob[0], flip_prob[0]]
        )
        if enable:
            points[:, 1] = -points[:, 1]

            for gt_boxes in gt_boxes_list:
                gt_boxes[:, 1] = -gt_boxes[:, 1]
                gt_boxes[:, -1] = -gt_boxes[:, -1] 
                if gt_boxes.shape[1] > 7:  #  x, y, z, l, w, h, vx, vy, rz
                    gt_boxes[:, 7] = -gt_boxes[:, 7] 

    # y flip
    if flip_prob[1] > 0:
        enable = np.random.choice(
            [False, True], replace=False, p=[1 - flip_prob[1], flip_prob[1]])
        if enable:
            points[:, 0] = -points[:, 0]
            
            for gt_boxes in gt_boxes_list:
                gt_boxes[:, 0] = -gt_boxes[:, 0]
                gt_boxes[:, -1] = -gt_boxes[:, -1] + np.pi  

                if gt_boxes.shape[1] > 7:   #  x, y, z, l, w, h, vx, vy, rz
                    gt_boxes[:, 6] = -gt_boxes[:, 6]
    
    for gt_boxes in gt_boxes_list:
        cond = gt_boxes[:, -1] > np.pi
        gt_boxes[cond, -1] = gt_boxes[cond, -1] - 2 * np.pi
        
        cond = gt_boxes[:, -1] < -np.pi
        gt_boxes[cond, -1] = gt_boxes[cond, -1] + 2 * np.pi

    return gt_boxes_list, points


def global_scaling(gt_boxes_list, points, min_scale=0.95, max_scale=1.05):
    noise_scale = np.random.uniform(min_scale, max_scale)
    points[:, :3] *= noise_scale
    for gt_boxes in gt_boxes_list:
        gt_boxes[:, :-1] *= noise_scale
    return gt_boxes_list, points

def global_rotation(gt_boxes_list, points, rotation=np.pi / 4):
    if not isinstance(rotation, list):
        rotation = [-rotation, rotation]
    noise_rotation = np.random.uniform(rotation[0], rotation[1])
    
    points[:, :3] = box_np_ops.yaw_rotation(points[:, :3], noise_rotation)
    for gt_boxes in gt_boxes_list:
        gt_boxes[:, :3] = box_np_ops.yaw_rotation(gt_boxes[:, :3], noise_rotation)
        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 6:8] = box_np_ops.yaw_rotation(np.hstack([gt_boxes[:, 6:8], np.zeros((gt_boxes.shape[0], 1))]),
                noise_rotation)[:, :2]
    
        gt_boxes[:, -1] += noise_rotation

    return gt_boxes_list, points

def global_translate_(gt_boxes, points, noise_translate_std):
    """
    Apply global translation to gt_boxes and points.
    """

    if not isinstance(noise_translate_std, (list, tuple, np.ndarray)):
        noise_translate_std = np.array(
            [noise_translate_std, noise_translate_std, noise_translate_std]
        )
    if all([e == 0 for e in noise_translate_std]):
        return gt_boxes, points
    noise_translate = np.array(
        [
            np.random.normal(0, noise_translate_std[0], 1),
            np.random.normal(0, noise_translate_std[1], 1),
            np.random.normal(0, noise_translate_std[0], 1),
        ]
    ).T

    points[:, :3] += noise_translate
    gt_boxes[:, :3] += noise_translate

    return gt_boxes, points

@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack(
        (boxes, boxes[:, slices, :]), axis=2
    )  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    # vec = np.zeros((2,), dtype=boxes.dtype)
    boxes_standup = box_np_ops.corner_to_standup_nd_jit(boxes)
    qboxes_standup = box_np_ops.corner_to_standup_nd_jit(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            iw = min(boxes_standup[i, 2], qboxes_standup[j, 2]) - max(
                boxes_standup[i, 0], qboxes_standup[j, 0]
            )
            if iw > 0:
                ih = min(boxes_standup[i, 3], qboxes_standup[j, 3]) - max(
                    boxes_standup[i, 1], qboxes_standup[j, 1]
                )
                if ih > 0:
                    for k in range(4):
                        for l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, l, 0]
                            D = lines_qboxes[j, l, 1]
                            acd = (D[1] - A[1]) * (C[0] - A[0]) > (C[1] - A[1]) * (
                                D[0] - A[0]
                            )
                            bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (
                                D[0] - B[0]
                            )
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (
                                    C[0] - A[0]
                                )
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (
                                    D[0] - A[0]
                                )
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        box_overlap_qbox = True
                        for l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (boxes[i, k, 0] - qboxes[j, l, 0])
                                cross -= vec[0] * (boxes[i, k, 1] - qboxes[j, l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for l in range(4):  # point l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (qboxes[j, k, 0] - boxes[i, l, 0])
                                    cross -= vec[0] * (qboxes[j, k, 1] - boxes[i, l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.
    return ret