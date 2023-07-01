'''
Author: Xiaolin Lin xlim1996@outlook.com
Date: 2023-05-19 19:21:26
LastEditors: Xiaolin Lin xlim1996@outlook.com
LastEditTime: 2023-05-31 17:37:21
FilePath: /detectron2/get_image.py
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

import pyzed.sl as sl


# Create a ZED camera
zed = sl.Camera()
# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
#TODO: change depth mode 
init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Use PERFORMANCE depth mode  
init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD1080 video mode
init_params.depth_minimum_distance = 0.2
init_params.camera_fps = 30  # Set fps at 30



# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)
calibration_params=zed.get_camera_information().camera_configuration.calibration_parameters
fx=calibration_params.left_cam.fx
fy=calibration_params.left_cam.fy
cx=calibration_params.left_cam.cx
cy=calibration_params.left_cam.cy
#capture image and depth from zed camera
image = sl.Mat()
depth = sl.Mat()
runtime_parameters = sl.RuntimeParameters()
i = 0
image_file_path="zed_image"
depth_file_path="zed_depth"
while(True):
        # Grab an image, a RuntimeParameters object must be given to grab()
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # A new image is available if grab() returns SUCCESS
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        image_save = image.get_data()
        depth_save = depth.get_data()

        # plt.imshow(depth_save,cmap='gray')
        # plt.show()
        cv2.imwrite(os.path.join(image_file_path,"image_{}.png".format(i)),image_save)
        # cv2.imwrite(os.path.join(file_path,"depth_{}.png".format(i)),depth_save)
        np.save(os.path.join(depth_file_path,"depth_{}.npy".format(i)),depth_save)
        input()
    i +=1

# Close the camera
        

        