'''
Author: Xiaolin Lin xlim1996@outlook.com
Date: 2023-03-24 15:07:21
LastEditors: Xiaolin Lin xlim1996@outlook.com
LastEditTime: 2023-04-10 14:06:28
FilePath: /detectron2/detectron2/data/datasets/__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# Copyright (c) Facebook, Inc. and its affiliates.
from .coco import load_coco_json, load_sem_seg, register_coco_instances, convert_to_coco_json
from .coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
from .lvis import load_lvis_json, register_lvis_instances, get_lvis_instances_meta
from .lvis_own import load_lvis_own_json, register_lvis_own_instances, get_lvis_own_instances_meta
from .pascal_voc import load_voc_instances, register_pascal_voc
from . import builtin as _builtin  # ensure the builtin datasets are registered


__all__ = [k for k in globals().keys() if not k.startswith("_")]
