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
from ne_eval import *
# Root directory of the project
ROOT_DIR = '../'

#piont names and class name
'''添加fashion ai'''
fi_class_names = ['dress']

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs/{}_logs".format(fi_class_names[0]))
model_path = os.path.join(ROOT_DIR, "model/mask_rcnn_{}.h5".format(fi_class_names[0]))

pre_result_save_path='../submit/pre_val_{}_result.csv'.format(fi_class_names[0])
gt_result_save_path='../submit/gt_val_{}_result.csv'.format(fi_class_names[0])
#result_save_path=('./data/test/{0}_result.csv'.format(fi_class_names[0]))


class_names_ = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left',
                   'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left',
                   'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in',
                   'cuff_right_out', 'top_hem_left', 'top_hem_right', 'waistband_left',
                   'waistband_right', 'hemline_left', 'hemline_right', 'crotch',
                   'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']
'''
各类存在的点在class_names_中的索引
'''
blouse_index=[0,1,2,3,4,5,6,9,10,11,12,13,14]#NUM_KEYPOINTS=13
skirt_index=[15,16,17,18]#NUM_KEYPOINTS=4
outwear_index=[0,1,3,4,5,6,7,8,9,10,11,12,13,14]#NUM_KEYPOINTS=14
dress_index=[0,1,2,3,4,5,6,7,8,9,10,11,12,17,18]#NUM_KEYPOINTS=15
trousers_index=[15,16,19,20,21,22,23]#NUM_KEYPOINTS=7


all_index={'blouse':blouse_index,
           'skirt':skirt_index,
           'outwear':outwear_index,
           'dress':dress_index,
           'trousers':trousers_index}
index = all_index[fi_class_names[0]]

fi_class_names_=[]
for i in index:
    fi_class_names_.append(class_names_[i])
print(fi_class_names_)

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
    NUM_KEYPOINTS = len(all_index[fi_class_names[0]])
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

###############################################################
class FIDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    """参数:category决定数据类别为train validation test"""

    def load_FI(self, category):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        train_data_path = '../data/train/'
        # Add classes
        for i, class_name in enumerate(fi_class_names):
            self.add_class("FI", i + 1, class_name)

        annotations = pd.read_csv('../data/train/Annotations/annotations.csv')
        annotations = annotations.append(pd.read_csv('../data/train/Annotations/train.csv'), ignore_index=True)
        annotations = annotations.append(pd.read_csv('../data/train/Annotations/test_a.csv'), ignore_index=True)
        annotations = annotations.append(pd.read_csv('../data/train/Annotations/test_b.csv'), ignore_index=True)
        annotations = annotations.append(pd.read_csv('../data/train/Annotations/data_scaling.csv'), ignore_index=True)
        annotations = annotations.append(pd.read_csv('../data/train/Annotations/data_flip_up_down.csv'), ignore_index=True)
        annotations = annotations.loc[annotations['image_category'] == fi_class_names[0]]
        annotations = annotations.reset_index(drop=True)  # 更新索引
        # 切分test数据集和train数据集
        np.random.seed(42)
        shuffled_indces = np.random.permutation(annotations.shape[0])
        val_set_size = int(annotations.shape[0] * 0.05)
        val_indices = shuffled_indces[:val_set_size]
        train_indices = shuffled_indces[val_set_size:]
        if category == 'train':
            annotations = annotations.iloc[train_indices]
        elif category == 'val':
            annotations = annotations.iloc[val_indices]
        else:
            # test 数据集
            pass
        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().

        annotations = annotations.reset_index(drop=True)  # 更新索引

        for x in range(annotations.shape[0]):
            # bg_color, shapes = self.random_image(height, width)
            id = annotations.loc[x, 'image_id']
            category = annotations.loc[x, 'image_category']
            print('loading image:%d/%d'%(x,annotations.shape[0]))
            im_path = os.path.join(train_data_path, id)

            # height, width = cv2.imread(im_path).shape[0:2]
            width, height = pic_height_width(im_path)

            key_points = []
            for key_point in annotations.loc[x, fi_class_names_].values:
                loc_cat = [int(j) for j in key_point.split('_')]
                key_points.append(loc_cat)

            self.add_image("FI", image_id=id, path=im_path,
                           width=width, height=height,
                           key_points=key_points, image_category=category)  # 添加我的数据

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

    def image_reference(self, image_id):
        """Return the key_points data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "FI":
            return info["key_points"], info["image_category"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        key_points = np.array(info['key_points'])
        clothing_nums = int(len(key_points) /config.NUM_KEYPOINTS)

        m = np.zeros([clothing_nums, info['height'], info['width'], config.NUM_KEYPOINTS])  # 生成24个mask,因为有24个关键点。

        class_mask = np.zeros([clothing_nums, config.NUM_KEYPOINTS])  # 点存在的状态经过处理有三种状态 不存在为0  1为不可见.2 为可见 三分类
        class_ids = []

        for clothing_num in range(clothing_nums):

            for part_num, bp in enumerate(key_points):
                if bp[2] > -1:  # AI数据编码为bp[2]=1为可见，=2为不可见，=3为不在图内或者不可推测，FI编码为=-1为不存在，0为不可见，1为可
                    m[clothing_num, bp[1], bp[0], part_num] = 1
                    class_mask[clothing_num, part_num] = bp[2] + 1
            class_ids.append(1)
        #class_ids=np.array([self.class_names.index(s[0]) for s in fi_class_names_])

        # Pach instance masks into an array
        if class_ids:
            mask = m
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids, class_mask
        else:
            return super(self.__class__).load_mask(image_id)

    def load_keypoints(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        image_category=info['image_category']
        key_points = np.array(info['key_points'])
        clothing_nums = int(len(key_points) / config.NUM_KEYPOINTS)
        keypoints = []
        keypoint = []
        class_ids = []
        for clothing_num in range(clothing_nums):

            for part_num, bp in enumerate(key_points):
                # if bp[2] > -1:  # AI数据编码为bp[2]=1为可见，=2为不可见，=3为不在图内或者不可推测，FI编码为=-1为不存在，0为不可见，1为可
                #     m[clothing_num, bp[1], bp[0], part_num] = 1
                #     class_mask[clothing_num, part_num] = bp[2] + 1
                if(bp[2]==-1):
                    keypoint += [0, 0, 0]
                else:
                    keypoint += [bp[0]-1,bp[1]-1,bp[2]+1]
            keypoint = np.reshape(keypoint,(-1,3))
            keypoints.append(keypoint)
            class_ids.append(1)
        #class_ids=np.array([self.class_names.index(image_category)])#如果多分类使用此处
        if class_ids:
            keypoints = np.array(keypoints, dtype=np.int32)
            class_ids = np.array(class_ids, dtype=np.int32)
            return keypoints, 0, class_ids
        else:
            return super(self.__class__).load_keypoints(image_id)


################################################################
'''
把int类型转为num_num_num格式以便提交
'''
def keypoint_to_str(keypoint):
    keypoint = keypoint.reshape([len(class_names_), 3])
    for x in range(len(class_names_)):
        if keypoint[x][2] != 1:
            keypoint[x] = [-1, -1, -1]
    list_keypoint = []
    for x in keypoint:
        list_keypoint.append(str(x[0]) + '_' + str(x[1]) + '_' + str(x[2]))
    return list_keypoint


'''
把得到的结果映射到24个点中。
'''
def keypoint_map_to24(points,img_category):
    x=[[-1,-1,-1] for i in range(24)]
    for point_index,x_index in enumerate(all_index[img_category]):
        #print(point_str_index)
        x[x_index]=points[point_index]
    return np.array(x)



if __name__ =='__main__':
    # Validation dataset
    dataset_val = FIDataset()
    dataset_val.load_FI(category='val')
    dataset_val.prepare()

    #config of model
    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    ###########################################################################
    #保存结果到csv
    ###########################################################################
    pre_point_to_csv_list=[]
    gt_point_to_csv_list=[]

    for x in range(0,dataset_val.num_images):
        image=dataset_val.load_image(x) #0为图像id
        category=dataset_val.image_info[x]['image_category'] #图像类别
        image_id=dataset_val.image_info[x]['id']

        results = model.detect_keypoint([image], verbose=0)

        r = results[0]  # for one image
        # log("image", image)
        # log("rois", r['rois'])
        # log("keypoints", r['keypoints'])
        # log("class_ids", r['class_ids'])
        # log("keypoints", r['keypoints'])
        error_count=0
        try:#统计未检测出目标的图片
            key_points = keypoint_map_to24(r['keypoints'][0], fi_class_names[0])
        except:
            key_points =np.array([[[0,0,0] for i in range(24)]])
            error_count+=1

        # visualize.display_keypoints(image,r['rois'],r['keypoints'], r['class_ids'], dataset_val.class_names)

        pre_point_str=keypoint_to_str(key_points)#把预测的坐标转为字符

        gt_point_str=keypoint_to_str(keypoint_map_to24(dataset_val.image_info[x]['key_points'],fi_class_names[0]))#真实的坐标转为字符

        # print('Clothes_test_eval:330',pre_point_str)
        # print('Clothes_test_eval:331',gt_point_str)

        pre_relust_info=[image_id,category]
        gt_relust_info=[image_id,category]

        pre_relust_info.extend(pre_point_str)
        gt_relust_info.extend(gt_point_str)

        pre_point_to_csv_list.append(pre_relust_info)
        gt_point_to_csv_list.append(gt_relust_info)

        print(error_count, r'/', x, r'/', dataset_val.num_images)

    '''
    保存结果
    '''
    columns=['image_id','image_category']#设置columns
    columns.extend(class_names_)      #

    pre_point_to_csv=pd.DataFrame(data=np.array(pre_point_to_csv_list).reshape([-1,26]),
                              columns=columns)
    gt_point_to_csv=pd.DataFrame(data=np.array(pre_point_to_csv_list).reshape([-1,26]),
                                 columns=columns)

    pre_point_to_csv.to_csv(pre_result_save_path,index=False)
    gt_point_to_csv.to_csv(gt_result_save_path,index=False)


    gt_data = read_data(gt_result_save_path)
    pre_data = read_data(pre_result_save_path)

    samples = len(gt_data.keys())
    norm = calculate_norm(gt_data)
    norm_dis,N,n_every_joints = calculate_norm_distance_mat(gt_data, pre_data,norm)
    print(norm_dis.shape)
    err = np.sum(norm_dis)/N
    print('err: ', err*100)

    err_joints = np.sum(norm_dis,axis=0)
    err_joints = np.divide(err_joints,n_every_joints)*100
    for i,v in enumerate(err_joints):
        print('joints '+str(i)+' mean err: ', v)
