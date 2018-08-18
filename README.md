# Mask RCNN for FashionAI Key Points Detection
-----------------------------------
*author1:huaifeng1993*  
*author2:sephirothhua* 
----------------------------------

从入学以来接触深度学习大约有半年时间，偶然得知天池大数据竞赛，便想尝试一下，从三月初报名，
然后找论文，找代码写代码，优化。最终排名67/2352，最终得分10左右%可能会进复赛。本次采用maskrcnn
进行点的预测，部分结果如下：  
<img src="https://github.com/huaifeng1993/FashionAI_key_point_location/blob/master/images/blouse.jpg" width="300" height="300" alt="blouse" align=center /><img src="https://github.com/huaifeng1993/FashionAI_key_point_location/blob/master/images/dress.jpg" width="300" height="300" alt="dress" align=center />
<img src="https://github.com/huaifeng1993/FashionAI_key_point_location/blob/master/images/outwear.jpg" width="300" height="300" alt="outwear" align=center /><img src="https://github.com/huaifeng1993/FashionAI_key_point_location/blob/master/images/skirt.jpg" width="300" height="300" alt="skirt" align=center />
<img src="https://github.com/huaifeng1993/FashionAI_key_point_location/blob/master/images/trousers.jpg" width="300" height="300" alt="trousers" align=center />
![b_final_reslut](https://github.com/huaifeng1993/FashionAI_key_point_location/blob/master/images/b_final_result.png)
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
## 2文件结构
```
project 
  |--README.md  
  |--data  
     |--round2test
     |--train   
        |--Images  
        |--Annotations  
           |--annotation.csv  
           |--train.csv   		   
           |--test_a.csv
           |--data_scaling.csv
           |--data_flip_up_down.csv
     |--val
        |--test_b.csv
  |--code   
  |--submit   
     |--类别_resutl_b.csv  
     |--test.csv  
  |--model  
     |--mask_rcnn_类别.h5  
  |--logs  
  ```
 ### 2.1文件目录说明
 * data存放训练数据，需要把比赛提供的两个训练集解压到同一个目录文件中。
 * data/round2test文件存放复赛数据集
 * data/train/Anotations存放原始标注和数据增强后的标注
 * code存放训练代码。测试代码。评价代码。
 * submit存放最终结果，和5个类分别预测的结果，验证集结果 。
 * model存放模型文件，包括基于coco训练集的预训模型[百度网盘下载链接](https://pan.baidu.com/s/12_4EPT6_E6dedNriA-ifeQ)
 * logs存放训练日志和训练过程中的模型
## 3.code文件中主要代码说明
   * single_train.py 主要训练代码。
   * Clothes_test.py 复赛测试代码。
   * DataVal_NE_Cal.py 计算验证集NE
   * data_cat.py 合并5个result.csv文件代码。(因为训练了5个模型，测试时候会分别产生5个结果。)
   * model.py 模型结构代码。
## 4.如何训练
   * 对5个类别分别训练，训练每一个类别时只需要在singnal_trian.py第36行修改fi_class_names，训练好的模型保存在
logs/类别_logs文件夹下。例如对blouse训练，只需要令fi_class_names = ['blouse']，训练其它四个类别同上操作。
## 5.如何测试
   * 把训练好的模型放在model文件夹下并更名为mask_rcnn_类别.h5，如mask_rcnn_blouse.h5
   * 因为训练了5个模型，所以在这里运行Clothes_test.py分别对5个类做预测，并输出5个 类别_resutl_b.csv文件
如blouse_result.csv。同样修改第23行fi_class_names = ['类别']，需要5个类别各运行一次，保存在'../submit文件夹下。
   * 最后运行dat_cat.py把5个结果合并成一个并命名为test.csv保存在‘../submit’文件夹下。
## 6.结果可视化
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


## 7.下载
  models:[BaiduCloud](https://pan.baidu.com/s/12_4EPT6_E6dedNriA-ifeQ)把模型下载放在model文件夹下。
  dataset:[BaiduCloud](https://pan.baidu.com/s/1mafQ8N9G1PReGpOgLM7LQw) 
## 感谢
  非常感谢  
  @Superlee506 [Mask_RCNN_Humanpose](https://github.com/Superlee506/Mask_RCNN_Humanpose)  
  @matterport [Mask_RCNN](https://github.com/matterport/Mask_RCNN)  
  @RodrigoGantier [Mask_R_CNN_Keypoints](https://github.com/RodrigoGantier/Mask_R_CNN_Keypoints)  
的maskrcnn代码。本人我的工作基于上述三人所贡献的代码展开。受益匪浅，膜拜大佬。
