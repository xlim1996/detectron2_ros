'''
Author: Xiaolin Lin xlim1996@outlook.com
Date: 2023-05-31 15:38:02
LastEditors: Xiaolin Lin xlim1996@outlook.com
LastEditTime: 2023-06-16 11:56:32
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
    parser.add_argument('--visualize_pcl',default=True, help='view the result or not')
    parser.add_argument('--image_dir', help='save the result or not')
    parser.add_argument('--depth_dir', help='save the result or not')
    parser.add_argument('--save_image',default=False, help='save the result or not')
    parser.add_argument('--save_image_path',default='result_image', help='save the result or not')
    parser.add_argument('--save_pcl',default=False, help='save the result or not')
    parser.add_argument('--save_pcl_path',default='result_pcl', help='save the result or not')
    parser.add_argument('--fx',default=5.242780151367187500e+02, help='save the result or not')
    parser.add_argument('--fy',default=5.242780151367187500e+02, help='save the result or not')
    parser.add_argument('--cx',default=6.233546142578125000e+02, help='save the result or not')
    parser.add_argument('--cy',default=3.711284484863281250e+02, help='save the result or not')
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
def detect(model,id,cfg,image,depth,fx,fy,cx,cy,visualize_image=True,visualize_pcl=True):
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
    image_pcd = o3d.geometry.Image(image_result)
    depth_pcd = o3d.geometry.Image(depth)
    #instance segmentation result to point cloud
    intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(width=W_o, height=H_o, fx=fx, fy=fy, cx=cx, cy=cy)

    rgbd_image=o3d.geometry.RGBDImage.create_from_color_and_depth(image_pcd,depth_pcd,depth_scale=1.0,depth_trunc=3.0,convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic_matrix)
    if visualize_pcl:
        o3d.visualization.draw_geometries([pcd])
    return image_result,pcd
def main():
    args = parse_args()
    config=args.config
    checkpoints=args.checkpoints
    image_path = args.image_path
    depth_path = args.depth_path
    image_dir = args.image_dir
    depth_dir = args.depth_dir

    #load the model
    model,id,cfg = load_model(config,checkpoints)
    #camera intrinsic parameters
    fx=args.fx
    fy=args.fy
    cx=args.cx
    cy=args.cy
    
    with torch.inference_mode():
        if image_dir is None:
            file_index = os.path.basename(image_path).split('.')[0].split('_')[1]
            print(image_path)
            print(file_index)
            image = cv2.imread(image_path)
            depth = np.load(depth_path)
            image_result,pcd=detect(model,id,cfg,image,depth,fx,fy,cx,cy,args.visualize_image,args.visualize_pcl)
            if args.save_image:
                save_image_path = os.path.join(args.save_image_path)
                if not os.path.exists(args.save_image_path):
                    os.makedirs(args.save_image_path)
                cv2.imwrite(os.path.join(save_image_path,os.path.basename(image_path)),image_result)
            if args.save_pcl:
                save_pcl_path = os.path.join(args.save_pcl_path)
                if not os.path.exists(save_pcl_path):
                    os.makedirs(save_pcl_path)
                o3d.io.write_point_cloud(os.path.join(save_pcl_path,"pcd_{}.ply".format(file_index)),pcd)
        else:
            image_list = os.listdir(image_dir)
            depth_list = os.listdir(depth_dir)
            assert len(image_list)==len(depth_list)
            for file_name in image_list:
                file_index = os.path.basename(file_name).split('.')[0].split('_')[1]
                image = cv2.imread(os.path.join(image_dir,file_name))
                depth = np.load(os.path.join(depth_dir,"depth_{}.npy".format(file_index)))   
                #if save the result,it won't visualize the result(avoid bugs)             
                image_result,pcd=detect(model,id,cfg,image,depth,fx,fy,cx,cy,False,False)
                if args.save_image:
                    save_image_path = os.path.join(args.save_image_path)
                    if not os.path.exists(args.save_image_path):
                        os.makedirs(args.save_image_path)
                    cv2.imwrite(os.path.join(save_image_path,file_name),image_result)
                if args.save_pcl:
                    save_pcl_path = os.path.join(args.save_pcl_path)
                    if not os.path.exists(save_pcl_path):
                        os.makedirs(save_pcl_path)
                    o3d.io.write_point_cloud(os.path.join(save_pcl_path,"pcd_{}.ply".format(file_index)),pcd)
if __name__ == '__main__':
    main()

