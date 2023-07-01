'''
Author: Xiaolin Lin xlim1996@outlook.com
Date: 2023-05-19 18:36:06
LastEditors: Xiaolin Lin xlim1996@outlook.com
LastEditTime: 2023-06-16 11:49:27
FilePath: /detectron2/try_zed_camera.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import time
import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d



from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer,GenericMask
from detectron2.structures.instances import Instances


def cv2_imshow(img_bgr):
    plt.rcParams['figure.figsize'] = (18, 36)
    plt.axis('off')
    plt.title('result')
    plt.imshow(img_bgr[...,::-1])
    plt.show()


cfg = LazyConfig.load('projects/ViTDet/configs/LVIS_MIX/cascade_mask_rcnn_vitdet_h_100ep.py')
metadata = MetadataCatalog.get(cfg.dataloader.train.dataset.names) # to get labels from ids
classes = metadata.thing_classes

id=[]
for i in range(len(classes)):
    id.append(metadata.get('class_image_count')[i]['id'])
cfg = LazyConfig.load('projects/ViTDet/configs/LVIS/cascade_mask_rcnn_vitdet_h_100ep.py')
cfg.train.init_checkpoint = "checkpoints/ViTDet/model_final_11bbb7.pkl" # replace with the path were you have your model


model = instantiate(cfg.model)
model.to(cfg.train.device)
model = create_ddp_model(model)
DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

model.eval()
save = True
#camera intrinsic parameters
fx=5.242780151367187500e+02
fy=5.242780151367187500e+02
cx=6.233546142578125000e+02
cy=3.711284484863281250e+02
image_path = "zed_image/image_0.png"
image = cv2.imread(image_path)
depth_path = "zed_depth/depth_0.npy"
depth = np.load(depth_path)

with torch.inference_mode():
    print('image shape', image.shape)
    # input()
    H_o, W_o = image.shape[:2]
    image = cv2.resize(image,(int(W_o / 2), int(H_o / 2)))
    #convert data type of image
    image = np.array(image, dtype=np.uint8)
    H, W = image.shape[:2]
    image_model = np.moveaxis(image, -1, 0)        
    time_start = time.time()
    output = model([{'image': torch.from_numpy(image_model)}])
    time_end = time.time()
    print('time cost of loading the model', time_end - time_start, 's')
    time_start = time.time()
    result=[]
    #filter the result by id
    if "instances" in output[0]:
        instances = output[0]["instances"].to("cpu")
        for i in range(len(instances.pred_classes)):
            if instances.pred_classes[i]+1 in id:
                result.append(instances[i])
        result=Instances.cat(result)
    time_end = time.time()
    print('time cost of detection', time_end - time_start, 's')
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.dataloader.train.dataset.names), scale=1.2)
    # v = Visualizer(image_show[:, :, ::-1], metadata, scale=1.2)
    v = v.draw_instance_predictions(result.to("cpu"))
    output = v.get_image()[:, :, ::-1]
    #visualize the result
    cv2_imshow(output)
    #resize image back to original size
    image_result = cv2.resize(output, (W_o, H_o))
    image_result = o3d.geometry.Image(image_result)
    depth = o3d.geometry.Image(depth)
    #instance segmentation result to point cloud
    intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(width=W_o, height=H_o, fx=fx, fy=fy, cx=cx, cy=cy)

    rgbd_image=o3d.geometry.RGBDImage.create_from_color_and_depth(image_result,depth,depth_scale=1.0,depth_trunc=3.0,convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic_matrix)
    o3d.visualization.draw_geometries([pcd]) 

        

        