#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import Image as ROSImage  # 修改导入的数据类型
from cv_bridge import CvBridge
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

class Detectron2ROSNode:
    def __init__(self):

        # self.image_subscriber = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        self.image_publisher = rospy.Publisher('/object_detection/image', ROSImage, queue_size=10)
        self.cv_bridge = CvBridge()
        self.model,self.id,self.cfg = self.load_model()
        self.visualize_image = True
        self.rate = rospy.Rate(10) # 10hz
        self.camera = cv2.VideoCapture(0)  # 打开相机设备
        self.camera.set(cv2.CAP_PROP_FPS, 0.1)  # 设置相机帧率为 10fps

    # 其他代码部分省略

    def load_model(self):
        #load the model
        config_model='src/detectron2_ros/projects/ViTDet/configs/LVIS/cascade_mask_rcnn_vitdet_h_100ep.py'
        config_class = 'src/detectron2_ros/projects/ViTDet/configs/LVIS_MIX/cascade_mask_rcnn_vitdet_h_100ep.py'
        checkpoints='src/detectron2_ros/checkpoints/ViTDet/model_final_11bbb7.pkl'
        cfg = LazyConfig.load(config_class)
        metadata = MetadataCatalog.get(cfg.dataloader.train.dataset.names) # to get labels from ids
        classes = metadata.thing_classes
        #get the id of the classes you want to detect
        id=[]
        for i in range(len(classes)):
            id.append(metadata.get('class_image_count')[i]['id'])
        cfg = LazyConfig.load(config_model)
        cfg.train.init_checkpoint = checkpoints # replace with the path were you have your model


        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

        model.eval()
        return model,id,cfg
    def detect(self,image):
        with torch.inference_mode():
            H_o, W_o = image.shape[:2]
            #resize the image if it is too large
            if H_o >=1500 or W_o >=1500:
                image = cv2.resize(image,(int(W_o / 2), int(H_o / 2)))
            #convert data type of image
            image = np.array(image, dtype=np.uint8)
            image_model = np.moveaxis(image, -1, 0) 
            # time_start = time.time()
            output = self.model([{'image': torch.from_numpy(image_model)}])
            # time_end = time.time()
            # print('time cost of loading the model', time_end - time_start, 's')
            time_start = time.time()
            result=[]
            #filter the result by id
            if "instances" in output[0]:
                instances = output[0]["instances"].to("cpu")
                for i in range(len(instances.pred_classes)):
                    if instances.pred_classes[i]+1 in self.id:
                        result.append(instances[i])
                result=Instances.cat(result)
            time_end = time.time()
            print('time cost of detection', time_end - time_start, 's')
            v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.dataloader.train.dataset.names), scale=1.2)
            # v = Visualizer(image_show[:, :, ::-1], metadata, scale=1.2)
            v = v.draw_instance_predictions(result.to("cpu"))
            output = v.get_image()[:, :, ::-1]
            #visualize the result
            if self.visualize_image:
                self.cv2_imshow(output)
            return output

    def cv2_imshow(self,img_bgr):
        plt.rcParams['figure.figsize'] = (18, 36)
        plt.axis('off')
        plt.title('result')
        plt.imshow(img_bgr[...,::-1])
        plt.show()
    def run(self):
        rospy.loginfo('Start read')
        cap = cv2.VideoCapture(0)  # 0 represents the default camera
        while not rospy.is_shutdown():
            ret, frame = self.camera.read()
            if ret:
                output_image = self.detect(frame)
                ros_image = self.cv_bridge.cv2_to_imgmsg(output_image, "bgr8")
                self.image_publisher.publish(ros_image)
            self.rate.sleep()
        cap.release()
        rospy.loginfo('Done read')

def main(argv):
    rospy.init_node('detectron2_ros')
    node = Detectron2ROSNode()
    node.run()

if __name__ == '__main__':
    main(sys.argv)