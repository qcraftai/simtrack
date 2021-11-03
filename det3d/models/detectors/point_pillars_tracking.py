from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import time
import torch

@DETECTORS.register_module
class PointPillarsTracking(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(PointPillarsTracking, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

    def extract_feat(self, data):
        input_features = self.reader(data["features"], data["num_voxels"], data["coors"])
        
        x = self.backbone(input_features, data["coors"], data["batch_size"], data["input_shape"])

        if 'prev_hm' in data:
            prev_hm = torch.cat(data['prev_hm'], dim=1)
            x = torch.cat((x, prev_hm), dim=1)

        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        preds = self.bbox_head(x)
        
        if return_loss:
            curr_voxels = example["curr_voxels"]
            curr_coordinates = example["curr_coordinates"]
            curr_num_points_in_voxel = example["curr_num_points"]

            curr_data = dict(
                features=curr_voxels,
                num_voxels=curr_num_points_in_voxel,
                coors=curr_coordinates,
                batch_size=batch_size,
                input_shape=example["shape"][0],
            )

            curr_x = self.extract_feat(curr_data)
            curr_preds = self.bbox_head(curr_x)
        
        return_feature = kwargs.get('return_feature', False)
        if return_feature:
            return preds
        if return_loss:
            return self.bbox_head.loss_tracking(example, preds, curr_preds)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)
