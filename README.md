# detectron2_ros
This is a detecron2 ros package that uses VitDet as an instance segmentation method.
# How to use detectron2_ros:
  ## 1. Please follow the following link to install ROS(noetic).
    http://wiki.ros.org/noetic/Installation/Ubuntu
  ## 2. Git clone the repository
    git clone git@github.com:xlim1996/detectron2.git
    cd detectron2_ros  
  ## 3. Install conda environment:
    conda env create -f env.yml
    conda activate detectron2_ros
    cd src/detectron2_ros
    pip install -e .
    pip install pyzed
  ## 3. Download the checkpoints and put it into src/detectron2_ros/checkpoints/ViTDet. Here is the link:
    https://dl.fbaipublicfiles.com/detectron2/ViTDet/LVIS/cascade_mask_rcnn_vitdet_h/332552778/model_final_11bbb7.pkl
  ### If you want to use a different model, please change the checkpoints path and config file in object_detection_node.py(ROS node + detectron2),vitdet_pcl.py and vitdet_image.py(detectron2 only)
  ## 4. Test the Environment
    cd src/detectron2_ros && python try_zed_camera.py
  ## 5. Initialize the ROS workspace 
    cd src && catkin_init_workspace
  ## 6. Build
    cd .. && catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
    source devel/setup.bash
  ## 7. Running the node
    roslaunch detectron2_ros detectron2_ros.launch
