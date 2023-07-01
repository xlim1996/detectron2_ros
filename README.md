# detectron2_ros
This is a detecron2 ros package that uses VitDet(LVIS with selected object classes) as an instance segmentation method. 
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
    pip install pyzed(If you need to used zed camera)
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
  ## 8. How Change selected object classes
  cd src/detectron2_ros/detectron2/data/datasets
  - There are four files you need to understand: lvis_own_categories.py, lvis_own_category_image_count.py, lvis_v1_categories.py, and lvis_v1_category_image_count.py.  
  - lvis_v1_categories.py contains all the object classes in the LVIS dataset, and lvis_v1_category_image_count.py contains the corresponding number of images for each class.  
  - lvis_own_categories.py includes the object classes we have selected, and lvis_own_category_image_count.py contains the respective number of images for those selected classes.
  - If you wish to change the selected object classes, you need to obtain the object classes from lvis_v1_categories.py and their corresponding image counts from lvis_v1_category_image_count.py. Then, copy this information to lvis_own_categories.py and lvis_own_category_image_count.py, respectively.
    
