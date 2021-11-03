import numba
import numpy as np
from det3d.core.bbox.geometry import (
    points_count_convex_polygon_3d_jit,
    points_in_convex_polygon_3d_jit,
)


def points_count_rbbox(points, rbbox, origin=(0.5, 0.5, 0.5)):
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, -1], origin=origin
    )
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    return points_count_convex_polygon_3d_jit(points[:, :3], surfaces)


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1
    ).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2 ** ndim, ndim])
    return corners


@numba.njit
def corners_2d_jit(dims, origin=0.5):
    ndim = 2
    corners_norm = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=dims.dtype)
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape((-1, 1, ndim)) * corners_norm.reshape((1, 2 ** ndim, ndim))
    return corners


@numba.njit
def corners_3d_jit(dims, origin=0.5):
    ndim = 3
    corners_norm = np.array(
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
        dtype=dims.dtype,
    ).reshape((8, 3))
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape((-1, 1, ndim)) * corners_norm.reshape((1, 2 ** ndim, ndim))
    return corners


@numba.njit
def corner_to_standup_nd_jit(boxes_corner):
    num_boxes = boxes_corner.shape[0]
    ndim = boxes_corner.shape[-1]
    result = np.zeros((num_boxes, ndim * 2), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(ndim):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(ndim):
            result[i, j + ndim] = np.max(boxes_corner[i, :, j])
    return result


def corner_to_standup_nd(boxes_corner):
    assert len(boxes_corner.shape) == 3
    standup_boxes = []
    standup_boxes.append(np.min(boxes_corner, axis=1))
    standup_boxes.append(np.max(boxes_corner, axis=1))
    return np.concatenate(standup_boxes, -1)

def limitperiod(val):
    offset=0.5
    period=np.pi*2
    return val - np.floor(val / period + offset) * period


def rotation_3d_in_axis(points, angles):  # corrected
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
   
    rot_mat_T = np.stack(
        [
            [rot_cos,  rot_sin, zeros],
            [-rot_sin, rot_cos, zeros],
            [zeros, zeros, ones],
        ]
    )

    return np.einsum("aij,jka->aik", points, rot_mat_T)


def yaw_rotation(points, yaw): # corrected 
    # points: [N, 3]
    rot_sin = np.sin(yaw)
    rot_cos = np.cos(yaw)
    
    rot_mat = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]], dtype=points.dtype)

    return points@rot_mat.T


def rotation_2d(points, angles):  # corrected, counterclock wise
    """rotation 2d points with given angle

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, rot_sin], [-rot_sin, rot_cos]])
    return np.einsum("aij,jka->aik", points, rot_mat_T)


def center_to_corner_box3d(centers, dims, angles=None, origin=(0.5, 0.5, 0.5)): # corrected
    """convert locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 3]): x, y, z
        dims (float array, shape=[N, 3]): size_x, size_y, size_z
        angles (float array, shape=[N]): yaw angle
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles)
    corners += centers.reshape([-1, 1, 3])
    return corners


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5): # corrected 
    """convert locations, dimensions and angles to corners.
    format: center(xy), dims(xy), yaw_angle

    Args:
        centers (float array, shape=[N, 2]): x, y
        dims (float array, shape=[N, 2]): size_x, size_y,
        angles (float array, shape=[N]): yaw angle

    Returns:
        [type]: [description]
    """
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners



def points_in_rbbox(points, rbbox, origin=(0.5, 0.5, 0.5)):
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, -1], origin=origin)
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return indices


def corner_to_surfaces_3d(corners):
    """convert 3d box corners from corner function above
    to surfaces that normal vectors all direct to internal.

    Args:
        corners (float array, [N, 8, 3]): 3d box corners.
    Returns:
        surfaces (float array, [N, 6, 4, 3]):
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    surfaces = np.array(
        [
            [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
            [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
            [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
            [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
            [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
            [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
        ]
    ).transpose([2, 0, 1, 3])
    return surfaces