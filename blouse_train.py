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
#import visualize
from model import log
from PIL import Image

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "blouse_logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "blouse_logs/mask_rcnn_fi_0087.h5")

'''添加fashion ai'''
fi_class_names_ = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left',
                   'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left',
                   'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in',
                   'cuff_right_out', 'top_hem_left', 'top_hem_right', 'waistband_left',
                   'waistband_right', 'hemline_left', 'hemline_right', 'crotch',
                   'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']
fi_class_names = ['blouse']


class FIConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "FI"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_KEYPOINTS = 24
    KEYPOINT_MASK_SHAPE = [56, 56]
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 24 key_point

    RPN_TRAIN_ANCHORS_PER_IMAGE = 150
    VALIDATION_STPES = 100
    STEPS_PER_EPOCH = 500
    MINI_MASK_SHAPE = (56, 56)
    KEYPOINT_MASK_POOL_SIZE = 7
            # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]
    WEIGHT_LOSS = True
    KEYPOINT_THRESHOLD = 0.005
    # Maximum number of ground truth instances to use in one image


config = FIConfig()


def pic_height_width(filepath):
    fp = open(filepath, 'rb')
    im = Image.open(fp)
    fp.close()
    x, y = im.size
    if(im.mode =='RGB'):
        return x,y
    else:
        return False,False

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
        train_data_path = './data/train/'
        # Add classes
        for i, class_name in enumerate(fi_class_names):
            self.add_class("FI", i + 1, class_name)

        annotations = pd.read_csv('./data/train/Annotations/annotations.csv')
        annotations = annotations.append(pd.read_csv('./data/train/Annotations/train.csv'), ignore_index=True)
        annotations = annotations.loc[annotations['image_category'] == fi_class_names[0]]
        #annotations = annotations.reset_index(drop=True)  # 更新索引
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
        clothing_nums = int(len(key_points) / 24)

        m = np.zeros([clothing_nums, info['height'], info['width'], 24])  # 生成24个mask,因为有24个关键点。

        class_mask = np.zeros([clothing_nums, 24])  # 点存在的状态经过处理有三种状态 不存在为0  1为不可见.2 为可见 三分类
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
        clothing_nums = int(len(key_points) / 24)
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


if __name__== '__main__':

    # Training dataset
    dataset_train = FIDataset()
    dataset_train.load_FI(category='train')
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FIDataset()
    dataset_val.load_FI(category='val')
    dataset_val.prepare()

    print("Classes: {}.\n".format(dataset_train.class_names))
    print("Train Images: {}.\n".format(len(dataset_train.image_ids)))
    print("Valid Images: {}".format(len(dataset_val.image_ids)))


    model = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)

    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last
    if init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)
    # Training - Stage 1
    print("Train heads")
    model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=100,
            layers='heads')
    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    '''
    print("Training Resnet layer 4+")
    model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=20,
            layers='4+')
    # Training - Stage 3
    # Finetune layers from ResNet stage 3 and up
    print("Training Resnet layer 3+")
    model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 100,
            epochs=100,
            layers='all')
    '''