import cv2
import pandas as pd
import os
import numpy as np
import csv
data_path = '../data/train/'

save_name = 'Images_scaling'
annotations_save_path = os.path.join(data_path,'Annotations')
save_path = os.path.join(data_path,save_name)

fi_class_names_ = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left',
                   'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left',
                   'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in',
                   'cuff_right_out', 'top_hem_left', 'top_hem_right', 'waistband_left',
                   'waistband_right', 'hemline_left', 'hemline_right', 'crotch',
                   'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']

annotations = pd.read_csv('../data/train/Annotations/test_a.csv')
annotations = annotations.append(pd.read_csv('../data/train/Annotations/annotations.csv'), ignore_index=True)
annotations = annotations.append(pd.read_csv('../data/train/Annotations/train.csv'), ignore_index=True)
# annotations = annotations.loc[annotations['image_category'] == fi_class_names]


if (os.path.exists(annotations_save_path))==False:
    os.makedirs(annotations_save_path)

if (os.path.exists(save_path))==False:
    os.makedirs(save_path)
    os.makedirs(os.path.join(save_path,'blouse'))
    os.makedirs(os.path.join(save_path,'trousers'))
    os.makedirs(os.path.join(save_path,'dress'))
    os.makedirs(os.path.join(save_path,'skirt'))
    os.makedirs(os.path.join(save_path,'outwear'))

# csvfile=open(annotations_save_path + '/data.csv',"w+")
# writer = csv.writer(csvfile)
# reader = csv.reader(csvfile)
# if(reader.line_num == 0):
#     writer.writerow(["image_id", "image_category"]+fi_class_names_)

'''
把int类型转为num_num_num格式以便提交
'''
def keypoint_to_str(keypoint):
    list_keypoint = []
    for x in keypoint:
        list_keypoint.append(str(x[0]) + '_' + str(x[1]) + '_' + str(x[2]))
    return list_keypoint


csv_all = []
for x in range(annotations.shape[0]):
    id = annotations.loc[x, 'image_id']
    category = annotations.loc[x, 'image_category']
    print('loading image:%d/%d' % (x, annotations.shape[0]))
    im_path = os.path.join(data_path, id)


    key_points = []
    for key_point in annotations.loc[x, fi_class_names_].values:
        loc_cat = [int(j) for j in key_point.split('_')]
        key_points.append(loc_cat)

    img = cv2.imread(im_path)

    re_size = np.random.uniform(0.5,0.9)
    img = cv2.resize(img, (0,0),fx=re_size,fy=re_size)

    for i in range(len(key_points)):
        if(key_points[i][2]!=-1):
            key_points[i][0] *= re_size
            key_points[i][1] *= re_size
            key_points[i][0] = int(key_points[i][0])
            key_points[i][1] = int(key_points[i][1])
            #屏蔽掉，不屏蔽会把点也加在图片中
            #cv2.rectangle(img=img,pt1 = (key_points[i][0]-2,key_points[i][1]-2),pt2=(key_points[i][0]+2,key_points[i][1]+2),thickness= 2,color = (255,0,0))
    key_points = keypoint_to_str(key_points)
    new_dir = id.replace("Images",save_name)
    relust_info = [new_dir, category] + key_points
    csv_all.append(relust_info)
    #cv2.imshow('11', img)
    cv2.imwrite(os.path.join(data_path,new_dir),img)
    #cv2.waitKey(0)

columns=['image_id','image_category']#设置columns
columns.extend(fi_class_names_)
point_to_csv=pd.DataFrame(data=np.array(csv_all).reshape([-1,26]),columns=columns)
point_to_csv.to_csv(annotations_save_path + '/data_scaling.csv',index=False)
# csvfile.close()

