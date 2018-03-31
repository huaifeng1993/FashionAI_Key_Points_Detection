import os

import model as modellib
from model import log
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "mylogs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Run one of the code blocks

# Shapes toy dataset
# import shapes
# config = shapes.ShapesConfig()

# MS COCO Dataset
import coco
config = coco.CocoConfig()
COCO_DIR = "D:/Github/FastMaskRCNN/data/coco"  # TODO: enter value here
# Load dataset
assert config.NAME == "coco"
# Training dataset
# load person keypoints dataset
train_dataset_keypoints = coco.CocoDataset(task_type="person_keypoints")
train_dataset_keypoints.load_coco(COCO_DIR, "train")
train_dataset_keypoints.prepare()

val_dataset_keypoints = coco.CocoDataset(task_type="person_keypoints")
val_dataset_keypoints.load_coco(COCO_DIR, "val")
val_dataset_keypoints.prepare()

print("Train Keypoints Image Count: {}".format(len(train_dataset_keypoints.image_ids)))
print("Train Keypoints Class Count: {}".format(train_dataset_keypoints.num_classes))
for i, info in enumerate(train_dataset_keypoints.class_info):
    print("{:3}. {:50}".format(i, info['name']))

print("Val Keypoints Image Count: {}".format(len(val_dataset_keypoints.image_ids)))
print("Val Keypoints Class Count: {}".format(val_dataset_keypoints.num_classes))
for i, info in enumerate(val_dataset_keypoints.class_info):
    print("{:3}. {:50}".format(i, info['name']))

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO

model.load_weights(COCO_MODEL_PATH, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
# model.load_weights(model.find_last()[1],by_name=True)
# model.keras_model.summary()

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.

# Training - Stage 1
print("Train heads")
model.train(train_dataset_keypoints, val_dataset_keypoints,
            learning_rate=config.LEARNING_RATE,
            epochs=15,
            layers='heads')
# Training - Stage 2
# Finetune layers from ResNet stage 4 and up
print("Training Resnet layer 4+")
model.train(train_dataset_keypoints, val_dataset_keypoints,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=20,
            layers='4+')
# Training - Stage 3
# Finetune layers from ResNet stage 3 and up
print("Training Resnet layer 3+")
model.train(train_dataset_keypoints, val_dataset_keypoints,
            learning_rate=config.LEARNING_RATE / 100,
            epochs=100,
            layers='all')