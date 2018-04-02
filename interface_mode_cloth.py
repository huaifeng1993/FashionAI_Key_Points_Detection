import os
import random
import numpy as np
from config import Config
import model as modellib
import visualize
from model import log
from Clothes_train import FIDataset
from PIL import Image
# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed

# Get path to saved weights
# Either set a specific path or find last trained weights
model_path = os.path.join(ROOT_DIR, "mask_rcnn_fi_0080.h5")

class InferenceConfig(Config):
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

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Load dataset
assert inference_config.NAME == "FI"

# Training dataset
# load person keypoints dataset
# Training dataset
dataset_train = FIDataset()
dataset_train.load_FI(category='train')
dataset_train.prepare()

# Validation dataset
dataset_val = FIDataset()
dataset_val.load_FI(category='val')
dataset_val.prepare()

# Test on a random image############################################################################
image_id = random.choice(dataset_val.image_ids)
print(image_id)
# image_id = 141
original_image, image_meta, gt_class_id, gt_bbox, gt_keypoint =\
    modellib.load_image_gt_keypoints(dataset_val, inference_config,
                           image_id, augment=False,use_mini_mask=inference_config.USE_MINI_MASK)

class_name = dataset_val.image_info[image_id]['id'].split('/')[1]
print(class_name)
image_path = dataset_val.image_info[image_id]['id']
image_path = os.path.join(ROOT_DIR,'data','train',image_path)

###################################################################################################
log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_keypoint", gt_keypoint)
visualize.display_keypoints(original_image,gt_bbox,gt_keypoint,gt_class_id,dataset_val.class_names)
#####################################################################################################


def key_point_disappear(keypoint_list,box,class_name):
  keypoints = list()
  boxes = list()
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
      keypoint = skirt_keypoint(keypoint_list[0])
      box = box[0]
  elif (class_name=='outwear'):
    if (keypoint_list.shape[0] != 1):
      a = np.where(box[:, 0] == np.min(box[:, 0]))
      index = int(a[0])
      keypoint = outwear_keypoint(keypoint_list[index])
      box = box[index]
    else:
      keypoint = outwear_keypoint(keypoint_list[0])
      box = box[0]
  elif (class_name=='trousers'):
    if (keypoint_list.shape[0] != 1):
      a = np.where(box[:, 0] == np.max(box[:, 0]))
      index = int(a[0])
      keypoint = trousers_keypoint(keypoint_list[index])
      box = box[index]
    else:
      keypoint = trousers_keypoint(keypoint_list[0])
      box = box[0]
  keypoints.append(keypoint)
  boxes.append(box)
  return np.array(keypoints,dtype= np.int32),np.array(boxes,dtype= np.int32)

def blouse_keypoint(keypoint):
  keypoint[7:8,2]=0
  keypoint[15:,2]=0
  return keypoint

def dress_keypoint(keypoint):
  keypoint[13:16, 2] = 0
  keypoint[19:,2] = 0
  return keypoint

def skirt_keypoint(keypoint):
  keypoint[:14,2] = 0
  keypoint[19:,2] = 0
  return keypoint

def outwear_keypoint(keypoint):
  keypoint[2, 2] = 0
  keypoint[15:, 2] = 0
  return keypoint

def trousers_keypoint(keypoint):
  keypoint[:14, 2] = 0
  keypoint[17:18,2] = 0
  return keypoint

#Predict the keypoint######################################################################################
fp = open(image_path,'rb')
read_image = Image.open(fp)
image = np.asarray(read_image,dtype=np.uint8)
print(image_path)
results = model.detect_keypoint([image], verbose=0)
r = results[0] # for one image
log("rois",r['rois'])
log("keypoints",r['keypoints'])
log("class_ids",r['class_ids'])
log("keypoints",r['keypoints'])


keypoint,box = key_point_disappear(r['keypoints'],r['rois'],class_name)

print(keypoint)
visualize.key_point_draw(image,box,keypoint,r['class_ids'],dataset_val.class_names)


###########################################################################################################

