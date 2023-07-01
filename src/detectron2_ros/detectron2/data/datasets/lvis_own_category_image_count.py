'''
Author: Xiaolin Lin xlim1996@outlook.com
Date: 2023-04-02 12:55:22
LastEditors: Xiaolin Lin xlim1996@outlook.com
LastEditTime: 2023-04-17 16:25:13
FilePath: /hiwi/detectron2/detectron2/data/datasets/lvis_own_category_image_count.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# Copyright (c) Facebook, Inc. and its affiliates.
# Autogen with
# with open("lvis_v1_train.json", "r") as f:
#     a = json.load(f)
# c = a["categories"]
# for x in c:
# del x["name"]
# del x["instance_count"]
# del x["def"]
# del x["synonyms"]
# del x["frequency"]
# del x["synset"]
# LVIS_CATEGORY_IMAGE_COUNT = repr(c) + "  # noqa"
# with open("/tmp/lvis_category_image_count.py", "wt") as f:
#     f.write(f"LVIS_CATEGORY_IMAGE_COUNT = {LVIS_CATEGORY_IMAGE_COUNT}")
# Then paste the contents of that file below

# fmt: off
LVIS_CATEGORY_IMAGE_COUNT = [{'id': 2, 'image_count': 364}, {'id': 4, 'image_count': 149}, 
                             {'id': 12, 'image_count': 1207}, {'id': 19, 'image_count': 561},
                             {'id': 20, 'image_count': 8}, {'id': 23, 'image_count': 1883}, 
                             {'id': 27, 'image_count': 117}, {'id': 45, 'image_count': 1787}, 
                             {'id': 77, 'image_count': 1765}, {'id': 79, 'image_count': 125}, 
                             {'id': 83, 'image_count': 322}, {'id': 84, 'image_count': 60}, 
                             {'id': 87, 'image_count': 333}, {'id': 110, 'image_count': 1671}, 
                             {'id': 116, 'image_count': 104}, {'id': 127, 'image_count': 1903}, 
                             {'id': 128, 'image_count': 70}, {'id': 133, 'image_count': 1901}, 
                             {'id': 139, 'image_count': 1922}, {'id': 149, 'image_count': 17}, 
                             {'id': 150, 'image_count': 1567}, {'id': 154, 'image_count': 1309}, 
                             {'id': 170, 'image_count': 24}, {'id': 175, 'image_count': 158}, 
                             {'id': 181, 'image_count': 1659}, {'id': 182, 'image_count': 7}, 
                             {'id': 183, 'image_count': 834}, {'id': 192, 'image_count': 445}, 
                             {'id': 204, 'image_count': 1135}, {'id': 217, 'image_count': 1222}, 
                             {'id': 232, 'image_count': 1927}, {'id': 233, 'image_count': 8}, 
                             {'id': 266, 'image_count': 5}, {'id': 285, 'image_count': 592}, 
                             {'id': 303, 'image_count': 166}, {'id': 342, 'image_count': 236}, 
                             {'id': 344, 'image_count': 1521}, {'id': 350, 'image_count': 1890}, 
                             {'id': 351, 'image_count': 1240}, {'id': 367, 'image_count': 227}, 
                             {'id': 369, 'image_count': 106}, {'id': 390, 'image_count': 1942}, 
                             {'id': 391, 'image_count': 19}, {'id': 395, 'image_count': 115}, 
                             {'id': 421, 'image_count': 1451}, {'id': 438, 'image_count': 32}, 
                             {'id': 469, 'image_count': 1861}, {'id': 477, 'image_count': 128}, 
                             {'id': 498, 'image_count': 1920}, {'id': 510, 'image_count': 233}, 
                             {'id': 531, 'image_count': 13}, {'id': 540, 'image_count': 1886}, 
                             {'id': 609, 'image_count': 489}, {'id': 610, 'image_count': 10}, 
                             {'id': 613, 'image_count': 81}, {'id': 615, 'image_count': 1868}, 
                             {'id': 631, 'image_count': 1846}, {'id': 639, 'image_count': 341}, 
                             {'id': 656, 'image_count': 28}, {'id': 658, 'image_count': 338}, 
                             {'id': 674, 'image_count': 2}, {'id': 687, 'image_count': 999}, 
                             {'id': 698, 'image_count': 1402}, {'id': 708, 'image_count': 814}, 
                             {'id': 742, 'image_count': 23}, {'id': 751, 'image_count': 229}, 
                             {'id': 774, 'image_count': 71}, {'id': 776, 'image_count': 109}, 
                             {'id': 781, 'image_count': 339}, {'id': 782, 'image_count': 153}, 
                             {'id': 804, 'image_count': 1912}, {'id': 818, 'image_count': 1932},
                             {'id': 819, 'image_count': 50}, {'id': 836, 'image_count': 479}, 
                             {'id': 838, 'image_count': 307}, {'id': 881, 'image_count': 1096}, 
                             {'id': 923, 'image_count': 707}, {'id': 961, 'image_count': 1635}, 
                             {'id': 982, 'image_count': 1899}, {'id': 987, 'image_count': 1}, 
                             {'id': 1000, 'image_count': 1127}, {'id': 1018, 'image_count': 297}, 
                             {'id': 1025, 'image_count': 333}, {'id': 1050, 'image_count': 1860}, 
                             {'id': 1051, 'image_count': 56}, {'id': 1052, 'image_count': 1582}, 
                             {'id': 1068, 'image_count': 40}, {'id': 1069, 'image_count': 35}, 
                             {'id': 1070, 'image_count': 135}, {'id': 1110, 'image_count': 1149}, 
                             {'id': 1139, 'image_count': 1866}, {'id': 1154, 'image_count': 48}, 
                             {'id': 1155, 'image_count': 1855}, {'id': 1162, 'image_count': 630}, 
                             {'id': 1165, 'image_count': 7}, {'id': 1172, 'image_count': 114}, 
                             {'id': 1190, 'image_count': 941}, {'id': 1192, 'image_count': 26}, 
                             {'id': 1200, 'image_count': 52}, {'id': 1203, 'image_count': 81}]
  # noqa
# fmt: on
