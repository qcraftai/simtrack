import numpy as np
from det3d.ops.point_cloud.point_cloud_ops import points_to_voxel


class VoxelGenerator:
    def __init__(self, voxel_size, point_cloud_range, max_num_points=30, max_voxels=20000):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size

    def generate(self, points):
        return points_to_voxel(
            points=points,
            voxel_size=self._voxel_size,
            coors_range=self._point_cloud_range,
            max_points=self._max_num_points,
            max_voxels=self._max_voxels,
            reverse_index=True,
        )

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def voxel_shape(self):
        return self._voxel_shape

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points

    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size
