from det3d import torchie
import numpy as np
import torch
import pdb
from ..registry import PIPELINES


class DataBundle(object):
    def __init__(self, data):
        self.data = data


@PIPELINES.register_module
class Reformat(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, res, info):
        meta = res["metadata"]
        points = res["lidar"]["points"]
        voxels = res["lidar"]["voxels"]
       
        data_bundle = dict(
            metadata=meta,
            points=points,
            voxels=voxels["voxels"],
            shape=voxels["shape"],
            num_points=voxels["num_points"],
            num_voxels=voxels["num_voxels"],
            coordinates=voxels["coordinates"],
            ref_from_car=res['ref_from_car'],
            car_from_global=res['car_from_global'],
        )

        if res["mode"] == "train":
            data_bundle.update(res["lidar"]["targets"])

        if res["mode"] == "val":
            data_bundle.update(dict(metadata=meta,))

        return data_bundle, info
