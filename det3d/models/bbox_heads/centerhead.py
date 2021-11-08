
import torch
from torch import nn

import logging
from collections import defaultdict
import numpy as np

from det3d.core import box_torch_ops

from det3d.models.utils import kaiming_init
from det3d.models.utils.misc import Sequential

from ..registry import HEADS

import copy

from .loss import FastFocalLoss, RegLoss



class SepHead(nn.Module):
    def __init__(
        self,
        in_channels,
        heads,
        head_conv=64,
        final_kernel=1,
        bn=False,
        init_bias=-2.19,
        **kwargs,
    ):
        super(SepHead, self).__init__(**kwargs)

        self.heads = heads 
        for head in self.heads:
            classes, num_conv = self.heads[head]

            fc = Sequential()
            for i in range(num_conv-1):
                fc.add(nn.Conv2d(in_channels, head_conv,
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
                if bn:
                    fc.add(nn.BatchNorm2d(head_conv))
                fc.add(nn.ReLU())

            fc.add(nn.Conv2d(head_conv, classes,
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))    

            if 'hm' in head:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

            self.__setattr__(head, fc)
        

    def forward(self, x):
        ret_dict = dict()        
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict
    

@HEADS.register_module
class CenterHeadV2(nn.Module):
    def __init__(
        self,
        in_channels,
        tasks,
        weight,
        code_weights,
        common_heads,
        logger=None,
        init_bias=-2.19,
        share_conv_channel=64,
        num_hm_conv=2,
    ):
        super(CenterHeadV2, self).__init__()

        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.code_weights = code_weights 
        self.weight = weight  # weight between hm loss and loc loss

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()


        if not logger:
            logger = logging.getLogger("CenterHead")
        self.logger = logger

        logger.info(
            f"num_classes: {num_classes}"
        )

        # a shared convolution 
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, share_conv_channel,
            kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True)
        )

        self.tasks = nn.ModuleList()
        print("Use HM Bias: ", init_bias)

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(hm=(num_cls, num_hm_conv)))
            self.tasks.append(
                SepHead(share_conv_channel, heads, bn=True, init_bias=init_bias, final_kernel=3)
            )

        logger.info("Finish CenterHead Initialization")

    def forward(self, x, *kwargs):
        ret_dicts = []

        x = self.shared_conv(x)

        for task in self.tasks:
            ret_dicts.append(task(x))

        return ret_dicts

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        return y

    def loss(self, example, preds_dicts, **kwargs):
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict['hm'] = self._sigmoid(preds_dict['hm'])

            hm_loss = self.crit(preds_dict['hm'], example['hm'][task_id], example['ind'][task_id], example['mask'][task_id], example['cat'][task_id])

            target_box = example['anno_box'][task_id]
            # reconstruct the anno_box from multiple reg heads
            
            preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                        preds_dict['vel'], preds_dict['rot']), dim=1)  

            ret = {}
 
            # Regression loss for dimension, offset, height, rotation            
            box_loss = self.crit_reg(preds_dict['anno_box'], example['mask'][task_id], example['ind'][task_id], target_box)

            loc_loss = (box_loss*box_loss.new_tensor(self.code_weights)).sum()

            loss = hm_loss + self.weight*loc_loss

            ret.update({'loss': loss, 'hm_loss': hm_loss.detach().cpu(), 'loc_loss':loc_loss, 'loc_loss_elem': box_loss.detach().cpu(), 'num_positive': example['mask'][task_id].float().sum()})

            rets.append(ret)
        
        """convert batch-key to key-batch
        """
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)

        return rets_merged

    @torch.no_grad()
    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        """decode, nms, then return the detection result. Additionaly support double flip testing 
        """
        # get loss info
        rets = []
        metas = []

        double_flip = test_cfg.get('double_flip', False)

        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=preds_dicts[0]['hm'].dtype,
                device=preds_dicts[0]['hm'].device,
            )

        for task_id, preds_dict in enumerate(preds_dicts):
            # convert N C H W to N H W C 
            for key, val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 1).contiguous()

            batch_size = preds_dict['hm'].shape[0]

            if "metadata" not in example or len(example["metadata"]) == 0:
                meta_list = [None] * batch_size
            else:
                meta_list = example["metadata"]
                if double_flip:
                    meta_list = meta_list[:4*int(batch_size):4]

            batch_hm = torch.sigmoid(preds_dict['hm'].clone())

            batch_dim = torch.exp(preds_dict['dim'])

            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]
            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']


            batch_rot = torch.atan2(batch_rots, batch_rotc)

            batch, H, W, num_cls = batch_hm.size()

            batch_reg = batch_reg.reshape(batch, H*W, 2)
            batch_hei = batch_hei.reshape(batch, H*W, 1)

            batch_rot = batch_rot.reshape(batch, H*W, 1)
            batch_dim = batch_dim.reshape(batch, H*W, 3)
            batch_hm = batch_hm.reshape(batch, H*W, num_cls)

            ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
            ys = ys.view(1, H, W).repeat(batch, 1, 1).to(batch_hm.device).float()
            xs = xs.view(1, H, W).repeat(batch, 1, 1).to(batch_hm.device).float()

            xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
            ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

            xs = xs * test_cfg.out_size_factor * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
            ys = ys * test_cfg.out_size_factor * test_cfg.voxel_size[1] + test_cfg.pc_range[1]

            batch_vel = preds_dict['vel']

            batch_vel = batch_vel.reshape(batch, H*W, 2)
            batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_vel, batch_rot], dim=2)
         
            metas.append(meta_list)

            rets.append(self.post_processing(batch_box_preds, batch_hm, test_cfg, post_center_range)) 

        # Merge branches results
        ret_list = []
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores"]:
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k in ["label_preds"]:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = torch.cat([ret[i][k] for ret in rets])

            ret['metadata'] = metas[0][i]
            ret_list.append(ret)

        return ret_list 

    @torch.no_grad()
    def post_processing(self, batch_box_preds, batch_hm, test_cfg, post_center_range):
        batch_size = len(batch_hm)

        prediction_dicts = []
        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            hm_preds = batch_hm[i]

            scores, labels = torch.max(hm_preds, dim=-1)

            score_mask = scores > test_cfg.score_threshold
            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) \
                & (box_preds[..., :3] <= post_center_range[3:]).all(1)

            mask = distance_mask & score_mask 

            box_preds = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]

            boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]

            selected = box_torch_ops.rotate_nms_pcdet(boxes_for_nms, scores, 
                                thresh=test_cfg.nms.nms_iou_threshold,
                                pre_maxsize=test_cfg.nms.nms_pre_max_size,
                                post_max_size=test_cfg.nms.nms_post_max_size)

            selected_boxes = box_preds[selected]
            selected_scores = scores[selected]
            selected_labels = labels[selected]

            prediction_dict = {
                'box3d_lidar': selected_boxes,
                'scores': selected_scores,
                'label_preds': selected_labels
            }

            prediction_dicts.append(prediction_dict)

        return prediction_dicts 

    @torch.no_grad()
    def predict_tracking(self, example, preds_dicts, test_cfg, **kwargs):
        """decode, nms, then return the detection result.
        """
        rets = []
        metas = []
        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=preds_dicts[0]['hm'].dtype,
                device=preds_dicts[0]['hm'].device)

        prev_track_id = kwargs.get('prev_track_id', None)

        if prev_track_id is not None:
            track_rets = []
            prev_hm = kwargs['prev_hm']
            
        for task_id, preds_dict in enumerate(preds_dicts):
            new_obj = [{}]
            # convert N C H W to N H W C 
            for key, val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 1).contiguous()
            
            batch_size = preds_dict['hm'].shape[0]

            if "metadata" not in example or len(example["metadata"]) == 0:
                meta_list = [None] * batch_size
            else:
                meta_list = example["metadata"]

            ######################################################
            batch_hm = torch.sigmoid(preds_dict['hm'])
            batch_dim = torch.exp(preds_dict['dim'])
            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]
            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']

            batch_rot = torch.atan2(batch_rots, batch_rotc)

            batch, H, W, num_cls = batch_hm.size()

            batch_reg = batch_reg.reshape(batch, H*W, 2)
            batch_hei = batch_hei.reshape(batch, H*W, 1)
            batch_rot = batch_rot.reshape(batch, H*W, 1)
            batch_dim = batch_dim.reshape(batch, H*W, 3)
            batch_hm = batch_hm.reshape(batch, H*W, num_cls)
            
            ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
            ys = ys.float().unsqueeze(0).to(preds_dicts[0]['hm'].device)
            xs = xs.float().unsqueeze(0).to(preds_dicts[0]['hm'].device)
            xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
            ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

            xs = xs * test_cfg.out_size_factor * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
            ys = ys * test_cfg.out_size_factor * test_cfg.voxel_size[1] + test_cfg.pc_range[1]

            batch_vel = preds_dict['vel']

            batch_vel = batch_vel.reshape(batch, H*W, 2)
            batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_vel, batch_rot], dim=2)
        
            metas.append(meta_list)

            if prev_track_id is not None:
                tracking_batch_hm =  (batch_hm + prev_hm[task_id]) / 2.0  
                tracking = self.post_processing_tracking(batch_box_preds, tracking_batch_hm, prev_track_id[task_id], test_cfg, post_center_range) 
                
                for bit in range(len(tracking)):
                    cond = tracking[bit]['tracking_id'] == -1 # new obj
                    for tk in tracking[0].keys():
                        if tk != 'tracking_id':
                            new_obj[bit][tk] = tracking[bit][tk][cond]
                
                for bit in range(len(tracking)):
                    cond = tracking[bit]['tracking_id'] != -1
                    for tk in tracking[0].keys():
                        tracking[bit][tk] = tracking[bit][tk][cond]
                
                track_rets.append(tracking)
                    
            else:
                new_obj = self.post_processing(batch_box_preds, batch_hm, test_cfg, post_center_range)
            
            rets.append(new_obj)
            
        # Merge branches results
        ret_list = []
        num_samples = len(rets[0])
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores", "selected_id"]:
                    ret[k] = torch.cat([retss[i][k] for retss in rets])
                elif k in ["label_preds"]:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                        
                    ret[k] = torch.cat([retss[i][k] for retss in rets])

            ret['metadata'] = metas[0][i]
            ret_list.append(ret)

        if prev_track_id is not None:
            track_rets_list = []
            num_tracks = len(track_rets[0])
            for i in range(num_tracks):
                ret = {}
                for k in ['box3d_lidar', 'scores', 'label_preds', 'tracking_id']:
                    if k in ["box3d_lidar", "scores", 'tracking_id']:
                        ret[k] = torch.cat([retss[i][k] for retss in track_rets])
                    elif k in ["label_preds"]:
                        flag = 0
                        for j, num_class in enumerate(self.num_classes):
                            track_rets[j][i][k] += flag
                            flag += num_class
                        ret[k] = torch.cat([retss[i][k] for retss in track_rets])

                ret['metadata'] = metas[0][i]
                track_rets_list.append(ret)

            return ret_list, track_rets_list
        else:
            return ret_list
    
    @torch.no_grad()
    def post_processing_tracking(self, batch_box_preds, batch_hm, prev_tracking_id, test_cfg, post_center_range):
        batch_size = len(batch_hm)

        prediction_dicts = []
        for i in range(batch_size):
            box_preds = batch_box_preds[i].clone()
            hm_preds = batch_hm[i].clone()
            prev_id = prev_tracking_id[i]
            scores, labels = torch.max(hm_preds, dim=-1)
            prev_id = torch.gather(prev_id, dim=1, index=labels.unsqueeze(-1)).squeeze(-1)
            score_mask = scores > test_cfg.score_threshold
            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) \
                & (box_preds[..., :3] <= post_center_range[3:]).all(1)

            mask = distance_mask & score_mask 
            
            box_preds = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]
            prev_id = prev_id[mask]

            boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]
            selected = box_torch_ops.rotate_nms_pcdet(boxes_for_nms, scores, 
                                thresh=test_cfg.nms.nms_iou_threshold,
                                pre_maxsize=test_cfg.nms.nms_pre_max_size,
                                post_max_size=test_cfg.nms.nms_post_max_size)

            selected_boxes = box_preds[selected]
            selected_scores = scores[selected]
            selected_labels = labels[selected]
            selected_id = prev_id[selected]

            prediction_dict = {
                'box3d_lidar': selected_boxes,
                'scores': selected_scores,
                'label_preds': selected_labels,
                "tracking_id": selected_id,
            }

            prediction_dicts.append(prediction_dict)

        return prediction_dicts 
    




