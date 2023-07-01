<!--
 * @Author: Xiaolin Lin xlim1996@outlook.com
 * @Date: 2023-05-05 10:49:10
 * @LastEditors: Xiaolin Lin xlim1996@outlook.com
 * @LastEditTime: 2023-05-31 19:05:11
 * @FilePath: /detectron2/README.md
 * @Description: 
-->
# How to use VitDet:
  ## 1. Git clone the repository
    git clone git@github.com:xlim1996/detectron2.git
    cd detectron2  
  ## 2. Install conda environment:
    conda env create -f environment.yml
    pip install -e .
    pip install pyzed
  ## 3. Download the checkpoints and put it into checkpoints/ViTDet. Here is the link:
    https://dl.fbaipublicfiles.com/detectron2/ViTDet/LVIS/cascade_mask_rcnn_vitdet_h/332552778/model_final_11bbb7.pkl
  ### If you want to use different model, please change the checkpoints path in vitdet_pcl.py and vitdet_image.py
  ## 4. Test the Environment
    cd detectron2 && python try_zed_camera.py
  ## 5. Use vitdet_pcl.py to generate segmentation results and its corresponding point cloud.
  ### For one image you can run the following code to visualize the result:
        python vitdet_pcl.py --image_path your_image_path --depth_path your_depth_path
  ### If you want to save the result:
        python vitdet_pcl.py --image_path your_image_path --depth_path your_depth_path --save_image True --save_image_path your_save_image_path --save_pcl True --save_pcl_path your_save_pcl_path
  ### For multiple images, here we assume you put the images into one folder and depth into the other folder,and we only provide save the result option in this case:
        python vitdet_pcl.py --image_dir your_image_dir_path --image_dir your_image_dir_path --save_image True --save_image_path your_save_image_path --save_pcl True --save_pcl_path your_save_pcl_path 
  
  ## 6. Use vitdet_image.py to generate segmentation results of an image without pointcloud:
        python vitdet_image.py --image_path your_image_path --depth_path your_depth_path
  ### If you want to save the result or test multiple images, please follow the vitdet_pcl.py's way to use vitdet_image.py
  ## 7. Collect data
  ### Here we assume you use zed camera, and you can the following code to collect data:
        python collect_data.py
  ## 8. Intrinsic matrix
  ### The default setup of intrinsic matrix is the zed camera intrinsic matrix in our lab. You can change it based on your case.
