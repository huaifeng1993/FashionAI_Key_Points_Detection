Mask RCNN for Human Pose Estimation
-----------------------------------

The original code is from "https://github.com/matterport/Mask_RCNN" on Python 3, Keras, and TensorFlow. The code reproduce the work of "https://arxiv.org/abs/1703.06870" for human pose estimation.
This project aims to addressing the [issue#2][2]. 
When I start it, I refer to another project by [@RodrigoGantier][3] .
## However RodrigoGantier's project has the following problems:
*  It's codes have few comments and still use the oringal names from [@Matterport][4]'s project, which make the project hard to understand. 
*  When I trained this model, I found it's hard to converge as described in [issue#3][5].

## Requirements
* Python 3.5+
* TensorFlow 1.4+
* Keras 2.0.8+
* Jupyter Notebook
* Numpy, skimage, scipy, Pillow, cython, h5py
# Getting Started
* [inference_humanpose.ipynb][6] shows how to predict the keypoint of human using my trained model. It randomly chooses a image from the validation set. You can download pre-trained COCO weights for human pose estimation (mask_rcnn_coco_humanpose.h5) from the releases page (https://github.com/Superlee506/Mask_RCNN_Humanpose/releases).
* [train_humanpose.ipynb][7] shows how to train the model step by step. You can also use "python train_humanpose.py" to  start training.
* [inspect_humanpose.ipynb][8] visulizes the proposal target keypoints to check it's validity. It also outputs some innner layers to help us debug the model.

# Discussion
* I convert the joint coordinates into an integer label ([0, 56*56)), and use  `tf.nn.sparse_softmax_cross_entropy_with_logits` as the loss function. This refers to the original [Detectron code][9] which is key reason why my loss can converge quickly.
* If you still want to use the keypoint mask as output, you'd better adopt the modified loss function proposed by [@QtSignalProcessing][10] in [issue#2][11]. Because after crop and resize, the keypoint masks may hava more than one 1 values, and this will make the original soft_cross entropy_loss hard to converge.
* Althougth the loss converge quickly, the prediction results isn't as good as the oringal papers, especially for right or left shoulder, right or left knee, etc. I'm confused with it, so I release the code and any contribution or suggestion to this repository is welcome.


  [1]: https://github.com/Superlee506/Mask_RCNN_Human_Pose
  [2]: https://github.com/matterport/Mask_RCNN/issues/2
  [3]: https://github.com/RodrigoGantier/Mask_R_CNN_Keypoints
  [4]: https://github.com/matterport/Mask_RCNN
  [5]: https://github.com/RodrigoGantier/Mask_R_CNN_Keypoints/issues/3
  [6]: https://github.com/Superlee506/Mask_RCNN/blob/master/inference_humanpose.ipynb
  [7]: https://github.com/Superlee506/Mask_RCNN/blob/master/train_human_pose.ipynb
  [8]: https://github.com/Superlee506/Mask_RCNN/blob/master/inspect_humanpose.ipynb
  [9]: https://github.com/facebookresearch/Detectron/blob/master/lib/utils/keypoints.py
  [10]: https://github.com/QtSignalProcessing
  [11]: https://github.com/matterport/Mask_RCNN/issues/2
  
