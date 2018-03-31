import os
import sys
import random
import math
import re
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/mask_rcnn_coco.h5")

'''添加fashion ai'''
fi_class_names_ = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left',
                   'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left',
                   'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in',
                   'cuff_right_out', 'top_hem_left', 'top_hem_right', 'waistband_left',
                   'waistband_right', 'hemline_left', 'hemline_right', 'crotch',
                   'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']
fi_class_names = ['clothing']


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
    IMAGES_PER_GPU = 1
    NUM_KEYPOINTS = 24
    KEYPOINT_MASK_SHAPE = [56, 56]
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 24 key_point

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


config = FIConfig()


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

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

        # 切分test数据集和train数据集
        np.random.seed(42)
        shuffled_indces = np.random.permutation(annotations.shape[0])
        val_set_size = int(annotations.shape[0] * 0.01)
        val_indices = shuffled_indces[:val_set_size]
        train_indices = shuffled_indces[val_set_size:500]
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

        # Pach instance masks into an array
        if class_ids:
            mask = m
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids, class_mask
        else:
            return super(self.__class__).load_mask(image_id)

    def load_keypoints(self, image_id):
        # """Load person keypoints for the given image.
        #
        # Returns:
        # key_points: num_keypoints coordinates and visibility (x,y,v)  [num_person,num_keypoints,3] of num_person
        # masks: A bool array of shape [height, width, instance count] with
        #     one mask per instance.
        # class_ids: a 1D array of class IDs of the instance masks, here is always equal to [num_person, 1]
        # """
        # # If not a COCO image, delegate to parent class.
        # image_info = self.image_info[image_id]
        # if image_info["source"] != "coco":
        #     return super(CocoDataset, self).load_mask(image_id)
        #
        # keypoints = []
        # class_ids = []
        # instance_masks = []
        # annotations = self.image_info[image_id]["annotations"]
        # # Build mask of shape [height, width, instance_count] and list
        # # of class IDs that correspond to each channel of the mask.
        # for annotation in annotations:
        #     class_id = self.map_source_class_id(
        #         "coco.{}".format(annotation['category_id']))
        #     assert class_id == 1
        #     if class_id:
        #
        #         #load masks
        #         m = self.annToMask(annotation, image_info["height"],
        #                            image_info["width"])
        #         # Some objects are so small that they're less than 1 pixel area
        #         # and end up rounded out. Skip those objects.
        #         if m.max() < 1:
        #             continue
        #         # Is it a crowd? If so, use a negative class ID.
        #         if annotation['iscrowd']:
        #             # Use negative class ID for crowds
        #             class_id *= -1
        #             # For crowd masks, annToMask() sometimes returns a mask
        #             # smaller than the given dimensions. If so, resize it.
        #             if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
        #                 m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
        #         instance_masks.append(m)
        #         #load keypoints
        #         keypoint = annotation["keypoints"]
        #         keypoint = np.reshape(keypoint,(-1,3))
        #
        #         keypoints.append(keypoint)
        #         class_ids.append(class_id)


        # Pack instance masks into an array
        # if class_ids:
        #     keypoints = np.array(keypoints,dtype=np.int32)
        #     class_ids = np.array(class_ids, dtype=np.int32)
        #     masks = np.stack(instance_masks, axis=2)
        #     return keypoints, masks, class_ids
        # else:
        #     # Call super class to return an empty mask
        #     return super(CocoDataset, self).load_keypoints(image_id)
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]

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
        if class_ids:
            keypoints = np.array(keypoints, dtype=np.int32)
            class_ids = np.array(class_ids, dtype=np.int32)
            return keypoints, 0, class_ids
        else:
            return super(self.__class__).load_keypoints(image_id)



def plot_mask_points(dataset, config, model, filter=True, image_id=None):
    if not image_id:
        image_id = random.choice(dataset.image_ids)
    original_image, image_meta, gt_bbox, gt_mask = modellib.load_image_gt(dataset,
                                                                          config,
                                                                          image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    mrcnn = model.run_graph([original_image], [
        ("detections", model.keras_model.get_layer("mrcnn_detection").output),
        ("masks", model.keras_model.get_layer("mrcnn_mask").output),
        ("mask_classes", model.keras_model.get_layer("mrcnn_class_mask").output),
    ])
    det_ix = mrcnn['detections'][0, :, 4]
    det_count = np.where(det_ix == 0)[0][0]
    det_masks = mrcnn['masks'][0, :det_count, :, :, :]
    det_boxes = mrcnn['detections'][0, :det_count, :]
    det_mask_classes = np.argmax(mrcnn['mask_classes'][0, :det_count, :, :], axis=2)
    det_mask_classes = np.where(det_mask_classes == 0, np.ones_like(det_mask_classes), np.zeros_like(det_mask_classes))

    visualize.draw_boxes(original_image, refined_boxes=det_boxes[:, :4])
    _, ax = plt.subplots(5, 5)
    for i in range(5):
        ax[0, i].set_title(fi_class_names_[i])
        if filter:
            m = np.where(det_masks[0, :, :, i] > 0.8, det_masks[0, :, :, i], 0)
            m = np.where(m == m.max(), 1, 0)
            m = m * det_mask_classes[0, i]
        else:
            m = det_masks[0, :, :, i] * det_mask_classes[0, i]
        ax[0, i].imshow(m, interpolation='none')
    for i in range(5):
        ax[1, i].set_title(fi_class_names_[5 + i])
        if filter:
            m = np.where(det_masks[0, :, :, 5 + i] > 0.8, det_masks[0, :, :, 5 + i], 0)
            m = np.where(m == m.max(), 1, 0)
            m = m * det_mask_classes[0, 5 + i]
        else:
            m = det_masks[0, :, :, 5 + i] * det_mask_classes[0, 5 + i]
        ax[1, i].imshow(m, interpolation='none')
    for i in range(5):
        ax[2, i].set_title(fi_class_names_[10 + i])
        if filter:
            m = np.where(det_masks[0, :, :, 10 + i] > 0.8, det_masks[0, :, :, 10 + i], 0)
            m = np.where(m == m.max(), 1, 0)
            m = m * det_mask_classes[0, 10 + i]
        else:
            m = det_masks[0, :, :, 10 + i] * det_mask_classes[0, 10 + i]
        ax[2, i].imshow(m, interpolation='none')
    for i in range(5):
        ax[3, i].set_title(fi_class_names_[15 + i])
        if filter:
            m = np.where(det_masks[0, :, :, 15 + i] > 0.8, det_masks[0, :, :, 15 + i], 0)
            m = np.where(m == m.max(), 1, 0)
            m = m * det_mask_classes[0, 15 + i]
        else:
            m = det_masks[0, :, :, 15 + i] * det_mask_classes[0, 15 + i]
        ax[3, i].imshow(m, interpolation='none')
    for i in range(4):
        ax[4, i].set_title(fi_class_names_[20 + i])
        if filter:
            m = np.where(det_masks[0, :, :, 20 + i] > 0.8, det_masks[0, :, :, 20 + i], 0)
            m = np.where(m == m.max(), 1, 0)
            m = m * det_mask_classes[0, 20 + i]
        else:
            m = det_masks[0, :, :, 20 + i] * det_mask_classes[0, 20 + i]
        ax[4, i].imshow(m, interpolation='none')

    ax[4, 4].set_title('Real image')
    visualize.draw_boxes(original_image, refined_boxes=det_boxes[:1, :4], ax=ax[4, 4])

    # Plot the gt mask points
    _, axx = plt.subplots(5, 5)
    axx[4, 4].set_title('Real image')
    visualize.draw_boxes(original_image, refined_boxes=gt_bbox[:1, :4], masks=gt_mask,
                         ax=axx[4, 4])
    original_image, image_meta, gt_bbox, gt_mask = modellib.load_image_gt(dataset,
                                                                          config,
                                                                          image_id, use_mini_mask=True)
    for i in range(5):
        axx[0, i].set_title(fi_class_names_[i])
        axx[0, i].imshow(gt_mask[0, :, :, i], interpolation='none')
    for i in range(5):
        axx[1, i].set_title(fi_class_names_[5 + i])
        axx[1, i].imshow(gt_mask[0, :, :, 5 + i], interpolation='none')
    for i in range(5):
        axx[2, i].set_title(fi_class_names_[10 + i])
        axx[2, i].imshow(gt_mask[0, :, :, 10 + i], interpolation='none')
    for i in range(5):
        axx[3, i].set_title(fi_class_names_[15 + i])
        axx[3, i].imshow(gt_mask[0, :, :, 15 + i], interpolation='none')
    for i in range(4):
        axx[4, i].set_title(fi_class_names_[20 + i])
        axx[4, i].imshow(gt_mask[0, :, :, 20 + i], interpolation='none')
    plt.show()
if __name__== '__main__':

    # Training dataset
    dataset_train = FIDataset()
    dataset_train.load_FI(category='train')
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FIDataset()
    dataset_val.load_FI(category='val')
    dataset_val.prepare()

    #print("Classes: {}.\n".format(dataset_train.class_names))
    #print("Train Images: {}.\n".format(len(dataset_train.image_ids)))
    print("Valid Images: {}".format(len(dataset_val.image_ids)))

    DEVICE = "/gpu:0"
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)

    path_save = 'mask_rcnn_coco.h5'
    model.load_weights(COCO_MODEL_PATH,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"], by_name=True)

    # Training - Stage 1
    print("Train heads")
    model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=15,
            layers='heads')
    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
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
