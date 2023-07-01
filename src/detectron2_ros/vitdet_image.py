'''
Author: Xiaolin Lin xlim1996@outlook.com
Date: 2023-05-31 15:38:02
LastEditors: Xiaolin Lin xlim1996@outlook.com
LastEditTime: 2023-06-16 11:57:11
FilePath: /detectron2/vitdet_pcl.py
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
import argparse


from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer,GenericMask
from detectron2.structures.instances import Instances

def parse_args():
    parser = argparse.ArgumentParser(description='VitDet for PCL')
    parser.add_argument('--config', default='projects/ViTDet/configs/LVIS_MIX/cascade_mask_rcnn_vitdet_h_100ep.py',help='config file path')
    parser.add_argument('--checkpoints', default='checkpoints/ViTDet/model_final_11bbb7.pkl',help='model file path')
    parser.add_argument('--image_path',default='zed_image/image_0.png', help='image file path')
    parser.add_argument('--depth_path',default='zed_depth/depth_0.npy', help='depth file path')
    parser.add_argument('--visualize_image',default=True, help='view the result or not')
    parser.add_argument('--image_dir', help='save the result or not')
    parser.add_argument('--save_image',default=False, help='save the result or not')
    parser.add_argument('--save_image_path',default='result_image', help='save the result or not')
    args = parser.parse_args()
    return args

#visualize the result
def cv2_imshow(img_bgr):
    plt.rcParams['figure.figsize'] = (18, 36)
    plt.axis('off')
    plt.title('result')
    plt.imshow(img_bgr[...,::-1])
    plt.show()

def load_model(config,checkpoints):
    cfg = LazyConfig.load(config)
    metadata = MetadataCatalog.get(cfg.dataloader.train.dataset.names) # to get labels from ids
    classes = metadata.thing_classes
    #get the id of the classes you want to detect
    id=[]
    for i in range(len(classes)):
        id.append(metadata.get('class_image_count')[i]['id'])
    cfg = LazyConfig.load('projects/ViTDet/configs/LVIS/cascade_mask_rcnn_vitdet_h_100ep.py')
    cfg.train.init_checkpoint = checkpoints # replace with the path were you have your model


    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

    model.eval()
    return model,id,cfg
def detect(model,id,cfg,image,visualize_image=True):
    H_o, W_o = image.shape[:2]
    #resize the image if it is too large
    if H_o >=1500 or W_o >=1500:
        image = cv2.resize(image,(int(W_o / 2), int(H_o / 2)))
    #convert data type of image
    image = np.array(image, dtype=np.uint8)
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
    if visualize_image:
        cv2_imshow(output)
    #resize image back to original size
    image_result = cv2.resize(output, (W_o, H_o))
    return image_result
def main():
    args = parse_args()
    config=args.config
    checkpoints=args.checkpoints
    image_path = args.image_path
    image_dir = args.image_dir

    #load the model
    model,id,cfg = load_model(config,checkpoints)

    with torch.inference_mode():
        if image_dir is None:
            image = cv2.imread(image_path)
            image_result=detect(model,id,cfg,image,args.visualize_image)
            if args.save_image:
                save_image_path = os.path.join(args.save_image_path)
                if not os.path.exists(args.save_image_path):
                    os.makedirs(args.save_image_path)
                cv2.imwrite(os.path.join(save_image_path,os.path.basename(image_path)),image_result)
  
        else:
            image_list = os.listdir(image_dir)
            for file_name in image_list:
                image = cv2.imread(os.path.join(image_dir,file_name))          
                image_result=detect(model,id,cfg,image,args.visualize_image)
                if args.save_image:
                    save_image_path = os.path.join(args.save_image_path)
                    if not os.path.exists(args.save_image_path):
                        os.makedirs(args.save_image_path)
                    cv2.imwrite(os.path.join(save_image_path,file_name),image_result)
if __name__ == '__main__':
    main()

