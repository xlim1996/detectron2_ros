'''
Author: Xiaolin Lin xlim1996@outlook.com
Date: 2023-03-30 11:40:52
LastEditors: Xiaolin Lin xlim1996@outlook.com
LastEditTime: 2023-04-10 12:50:25
FilePath: /detectron2/projects/ViTDet/configs/LVIS_MIX/cascade_mask_rcnn_vitdet_l_50ep.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from detectron2.config import LazyCall as L
from detectron2.data.detection_utils import get_fed_loss_cls_weights
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import FastRCNNOutputLayers, FastRCNNConvFCHead, CascadeROIHeads

from .mask_rcnn_vitdet_l_100ep import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)

# arguments that don't exist for Cascade R-CNN
[model.roi_heads.pop(k) for k in ["box_head", "box_predictor", "proposal_matcher"]]

model.roi_heads.update(
    _target_=CascadeROIHeads,
    num_classes=100,
    box_heads=[
        L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[256, 256, 256, 256],
            fc_dims=[1024],
            conv_norm="LN",
        )
        for _ in range(3)
    ],
    box_predictors=[
        L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            box2box_transform=L(Box2BoxTransform)(weights=(w1, w1, w2, w2)),
            num_classes="${...num_classes}",
            test_score_thresh=0.02,
            test_topk_per_image=300,
            cls_agnostic_bbox_reg=True,
            use_sigmoid_ce=True,
            use_fed_loss=True,
            get_fed_loss_cls_weights=lambda: get_fed_loss_cls_weights(
                dataloader.train.dataset.names, 0.5
            ),
        )
        for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
    ],
    proposal_matchers=[
        L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
        for th in [0.5, 0.6, 0.7]
    ],
)
