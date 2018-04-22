# Mask RCNN for FashionAI key point location
-----------------------------------
## 1.环境要求：
    Python 3.4/3.5
    numpy
    scipy
    Pillow
    cython
    matplotlib
    scikit-image
    tensorflow>=1.3.0
    keras>=2.0.8
    opencv-python
    h5py
    imgaug
    IPython[all]
## 2.如何使用
   * 基coco的Mask_RCNN预训练模型 mask_rcnn_coco.h5进行迁移训练。
   * 对5个类别分别训练，训练每一个类别时只需要在singnal_trian.py修改fi_class_names，训练好的模型保存在
logs/类别_logs文件夹下。例如对blouse训练，只需要令fi_class_names = ['blouse']，训练其它四个类别同上操作。
   * 把训练好的模型放在model文件夹下并更名为mask_rcnn_类别.h5，如mask_rcnn_blouse.h5
   * 因为训练了5个模型，所以在这里运行Clothes_test_b.py分别对5个类做预测，并输出5个类别_resutl.csv文件
如blouse_result.csv。同样修改fi_class_names = ['类别']，需要5个类别各运行一次，保存在'../data/tem_result
文件夹下。
   * 最后运行dat_cat.py把5个结果合并成一个并命名为test.csv保存在‘../submit’文件夹下。

## 3.结果可视化
    在Clothes_test_b.py中把201行的这段代码，和213行代码的注释去掉，即可查看图片的点预测结果。
        # log("image", image)
        # log("rois", r['rois'])
        # log("keypoints", r['keypoints'])
        # log("class_ids", r['class_ids'])
        # log("keypoints", r['keypoints'])
        .
        .
        .
        # visualize.display_keypoints(image,r['rois'],r['keypoints'], r['class_ids'], dataset_test.class_names)

## 4.代码文件夹结构
  project<br>
  |--README.md<br>
  |--data<br>
  &ensp;&ensp;|--test<br>
  &ensp;&ensp;|--train<br>
  |--code<br>
  |--data_b<br>
  &ensp;&ensp;|--Images<br>
  &ensp;&ensp;|--tem_result<br>
  &ensp;&ensp;&ensp;&ensp;|--类别_result.csv<br>
  |--submit<br>
  &ensp;&ensp;|--test.csv<br>
  |--model<br>
  &ensp;&ensp;|--mask_rcnn_类别.h5<br>
  |--logs<br>

## 5.模型下载
    地址: [百度云下载](https://pan.baidu.com/s/12_4EPT6_E6dedNriA-ifeQ).
    把模型下载放在model文件夹下。

## 感谢
    非常感谢@Superlee506 @matterport @RodrigoGantier的maskrcnn代码。本人我的工作基于上述三人所贡献的代码展开。
 收益匪浅，膜拜大佬。