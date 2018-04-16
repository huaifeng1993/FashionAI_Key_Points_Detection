import os
import sys
import random
import math
import re
import time
import numpy as np
import matplotlib
import pandas as pd
import tensorflow as tf
from config import Config
import utils
import model as modellib
import visualize
from model import log
from PIL import Image

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

#piont names and class name
'''添加fashion ai'''
fi_class_names_ = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left',
                   'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left',
                   'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in',
                   'cuff_right_out', 'top_hem_left', 'top_hem_right', 'waistband_left',
                   'waistband_right', 'hemline_left', 'hemline_right', 'crotch',
                   'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']
blouse_index=[0,1,2,3,4,5,6,9,10,11,12,13,14]
skirt_index=[15,16,17,18]
outwear_index=[0,1,3,4,5,6,7,8,9,10,11,12,13,14]
dress_index=[0,1,2,3,4,5,6,7,8,9,10,11,12,17,18]
trousers[15,16,19,20,21,22,23]

fi_class_names = ['skirt']
#############################################
#
#############################################
def pic_height_width(filepath):
    fp = open(filepath, 'rb')
    im = Image.open(fp)
    fp.close()
    x, y = im.size
    if(im.mode =='RGB'):
        return x,y
    else:
        return False,False


class InferenceConfig(Config):
  # Give the configuration a recognizable name
    NAME = "FI"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_KEYPOINTS = 4
    KEYPOINT_MASK_SHAPE = [56, 56]
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # background + 24 key_point

    RPN_TRAIN_ANCHORS_PER_IMAGE = 150
    VALIDATION_STPES = 100
    STEPS_PER_EPOCH = 100
    MINI_MASK_SHAPE = (56, 56)
    KEYPOINT_MASK_POOL_SIZE = 7
            # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]
    WEIGHT_LOSS = True
    KEYPOINT_THRESHOLD = 0.005
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 128


    DETECTION_MAX_INSTANCES = 1

'''
test data set class
'''
class FITestDataset(utils.Dataset):
    def load_FI_test(self):
        test_data_path='./data/test/'
        # Add classes
        for i, class_name in enumerate(fi_class_names):
            self.add_class("FI", i + 1, class_name)
        annotations = pd.read_csv('./data/test/test.csv')
        annotations = annotations.loc[annotations['image_category'] == fi_class_names[0]]
        annotations = annotations.reset_index(drop=True)  # 更新索引

        for x in range(annotations.shape[0]):
            # bg_color, shapes = self.random_image(height, width)
            id = annotations.loc[x, 'image_id']
            category = annotations.loc[x, 'image_category']
            #print('loading image:%d/%d'%(x,annotations.shape[0]))
            im_path = os.path.join(test_data_path, id)
            # height, width = cv2.imread(im_path).shape[0:2]
            width, height = pic_height_width(im_path)
            self.add_image("FI", image_id=id, path=im_path,
                           width=width, height=height,
                            image_category=category)  # 添加我的数据
    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        根据image_id读取图片
        """
        info = self.image_info[image_id]
        # image = cv2.imread(info['path'])
        image = Image.open(info['path'])
        image = np.array(image)
        return image
############################滤点###################################
global error_count
error_count = 0
def key_point_disappear(keypoint_list,box,class_name):
  keypoints = list()
  boxes = list()
  try:
      if(class_name=='blouse'):
        if (keypoint_list.shape[0] != 1):
          a = np.where(box[:, 0] == np.min(box[:, 0]))
          index = int(a[0])
          keypoint = blouse_keypoint(keypoint_list[index])
          box = box[index]
        else:
          keypoint = blouse_keypoint(keypoint_list[0])
          box = box[0]
      elif (class_name=='dress'):
        if (keypoint_list.shape[0] != 1):
          a = np.where(box[:, 0] == np.min(box[:, 0]))
          index = int(a[0])
          keypoint = dress_keypoint(keypoint_list[index])
          box = box[index]
        else:
          keypoint = dress_keypoint(keypoint_list[0])
          box = box[0]
      elif (class_name == 'skirt'):
        if (keypoint_list.shape[0] != 1):
          a = np.where(box[:, 0] == np.max(box[:, 0]))
          index = int(a[0])
          keypoint = skirt_keypoint(keypoint_list[index])
          box = box[index]
        else:
          box = box[0]
          keypoint = skirt_keypoint(keypoint_list[0])
      elif (class_name=='outwear'):
        if (keypoint_list.shape[0] != 1):
          a = np.where(box[:, 0] == np.min(box[:, 0]))
          index = int(a[0])
          box = box[index]
          keypoint = outwear_keypoint(keypoint_list[index],box)
        else:
          box = box[0]
          keypoint = outwear_keypoint(keypoint_list[0],box)
      elif (class_name=='trousers'):
        if (keypoint_list.shape[0] != 1):
          a = np.where(box[:, 0] == np.max(box[:, 0]))
          index = int(a[0])
          box = box[index]
          keypoint = trousers_keypoint(keypoint_list[index],box)
        else:
          box = box[0]
          keypoint = trousers_keypoint(keypoint_list[0],box)

      keypoints.append(keypoint)
      boxes.append(box)
      return np.array(keypoints,dtype= np.int32),np.array(boxes,dtype= np.int32)
  except:
      global error_count
      error_count+=1
      keypoint = np.empty((24, 3))
      keypoint[:, :] = -1
      box = np.empty((1, 4))
      box[:, :] = -1
      keypoints.append(keypoint)
      boxes.append(box)
      return np.array(keypoints, dtype=np.int32), np.array(boxes, dtype=np.int32)


def blouse_keypoint(keypoint):
  keypoint[7:9,2]=0
  keypoint[15:,2]=0
  return keypoint

def dress_keypoint(keypoint):
  keypoint[13:17, 2] = 0
  keypoint[19:,2] = 0
  return keypoint

def skirt_keypoint(keypoint):
    keypoint[:15,2] = 0
    keypoint[19:,2] = 0
    return keypoint

def outwear_keypoint(keypoint,box):
    keypoint[2, 2] = 0
    keypoint[15:, 2] = 0
    keypoint_sort = keypoint[0:2]#解决领子同点
    if(keypoint_sort[0][0]==keypoint_sort[1][0]):
        mid_box =  np.int32((keypoint[6][0]-keypoint[5][0])/2)
        mid_box += keypoint[5][0]
        if(keypoint_sort[0][0]>=mid_box):
            keypoint_sort[0][0] = box[1]+(mid_box-box[1]-(keypoint_sort[1][0]-mid_box))
        else:
            keypoint_sort[1][0] = box[3] - (mid_box - box[1] - (keypoint_sort[0][0] - mid_box))
        keypoint[0:2] = keypoint_sort

    keypoint_sort = keypoint[9:13]#解决袖口排序
    if (keypoint_sort[1][0] == keypoint_sort[3][0]):
        if (keypoint_sort[1][0] <= keypoint_sort[0][0]):
            keypoint_sort[3][0] = box[3]
        else:
            keypoint_sort[3][0] = box[1]
    keypoint_sort = keypoint_sort[np.lexsort(keypoint_sort[:, :-2].T)]
    keypoint[10] = keypoint_sort[0]
    keypoint[9] = keypoint_sort[1]
    keypoint[11] = keypoint_sort[2]
    keypoint[12] = keypoint_sort[3]
    return keypoint

def trousers_keypoint(keypoint,box):
  keypoint[:15, 2] = 0
  keypoint[17:19,2] = 0
  keypoint_sort = keypoint[20:25]
  if(keypoint_sort[1][0]==keypoint_sort[3][0]):
      if(keypoint_sort[1][0]<=keypoint_sort[0][0]):
          keypoint_sort[3][0] = box[3]
      else:
          keypoint_sort[3][0] = box[1]
  keypoint_sort = keypoint_sort[np.lexsort(keypoint_sort[:, :-2].T)]
  keypoint[21] = keypoint_sort[0]
  keypoint[20] = keypoint_sort[1]
  keypoint[22] = keypoint_sort[2]
  keypoint[23] = keypoint_sort[3]
  return keypoint


def skirt_to_24(keypoint):
    try:
        keypoint_real = np.ndarray(shape=(1,24,3),dtype=np.int32)
        keypoint_real.fill(0)
        keypoint_real[0,15:19] = keypoint[0]
        return keypoint_real
    except:
        global error_count
        error_count += 1
        keypoint_real = np.ndarray(shape=(1,24,3),dtype=np.int32)
        keypoint_real.fill(0)
        return keypoint_real
'''
把int类型转为num_num_num格式以便提交
'''
def keypoint_to_str(keypoint):
    keypoint = keypoint.reshape([24, 3])
    for x in range(24):
        if keypoint[x][2] != 1:
            keypoint[x] = [-1, -1, -1]
    list_keypoint = []
    for x in keypoint:
        list_keypoint.append(str(x[0]) + '_' + str(x[1]) + '_' + str(x[2]))
    return list_keypoint
#######################################################################
if __name__ =='__main__':
    dataset_test=FITestDataset()
    dataset_test.load_FI_test()
    dataset_test.prepare()

    #config of model
    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    model_path = os.path.join(ROOT_DIR, "skirt_mode/mask_rcnn_fi_0406.h5")
    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    ###########################################################################
    #保存结果到csv
    ###########################################################################
    point_to_csv_list=[]
    #trousers问题:
    #315
    #380
    for x in range(550,dataset_test.num_images):
        image=dataset_test.load_image(x) #0为图像id
        category=dataset_test.image_info[x]['image_category'] #图像类别
        image_id=dataset_test.image_info[x]['id']

        results = model.detect_keypoint([image], verbose=0)

        r = results[0]  # for one image
        # log("image", image)
        # log("rois", r['rois'])
        # log("keypoints", r['keypoints'])
        # log("class_ids", r['class_ids'])
        # log("keypoints", r['keypoints'])

        keypoint = skirt_to_24(r['keypoints'])
        # keypoint, box = key_point_disappear(keypoint, r['rois'], category)
        #
        #
        visualize.key_point_draw(image,r['rois'],keypoint, r['class_ids'], dataset_test.class_names)

        # visualize.key_point_draw(image, r['rois'], r['keypoints'], r['class_ids'], dataset_test.class_names)
        point_str=keypoint_to_str(keypoint)
        # point_str = keypoint_to_str(r['keypoints'][0])
        relust_info=[image_id,category]
        relust_info.extend(point_str)
        print(point_str)
        point_to_csv_list.append(relust_info)
        print(error_count,r'/',x,r'/',dataset_test.num_images)

    '''
    保存结果
    '''
    columns=['image_id','image_category']#设置columns
    columns.extend(fi_class_names_)      #

    point_to_csv=pd.DataFrame(data=np.array(point_to_csv_list).reshape([-1,26]),
                              columns=columns)
    point_to_csv.to_csv('./data/test/blouse_result.csv',index=False)
