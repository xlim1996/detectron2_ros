'''
Author: Xiaolin Lin xlim1996@outlook.com
Date: 2023-04-02 12:08:11
LastEditors: Xiaolin Lin xlim1996@outlook.com
LastEditTime: 2023-04-10 12:50:57
FilePath: /detectron2/projects/ViTDet/configs/LVIS_MIX/mask_rcnn_vitdet_b_100ep.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from detectron2.config import LazyCall as L
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.evaluation.lvis_evaluation import LVISEvaluator
from detectron2.data.detection_utils import get_fed_loss_cls_weights

from ..COCO.mask_rcnn_vitdet_b_100ep import (
    dataloader,
    model,
    train,
    lr_multiplier,
    optimizer,
)

dataloader.train.dataset.names = "lvis_v1_train_subdataset"
dataloader.train.sampler = L(RepeatFactorTrainingSampler)(
    repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
        dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.001
    )
)
dataloader.test.dataset.names = "lvis_v1_val_subdataset"
dataloader.evaluator = L(LVISEvaluator)(
    dataset_name="${..test.dataset.names}",
    max_dets_per_image=300,
)

model.roi_heads.num_classes = 100
model.roi_heads.box_predictor.test_score_thresh = 0.02
model.roi_heads.box_predictor.test_topk_per_image = 300
model.roi_heads.box_predictor.use_sigmoid_ce = True
model.roi_heads.box_predictor.use_fed_loss = True
model.roi_heads.box_predictor.get_fed_loss_cls_weights = lambda: get_fed_loss_cls_weights(
    dataloader.train.dataset.names, 0.5
)

# Schedule
# 100 ep = 156250 iters * 64 images/iter / 100000 images/ep
train.max_iter = 156250
train.eval_period = 30000

lr_multiplier.scheduler.milestones = [138889, 150463]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 250 / train.max_iter

optimizer.lr = 2e-4
