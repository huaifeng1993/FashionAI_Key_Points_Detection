import pandas as pd
import numpy as np
import os
'''
因为分开训练了5个模型，所以用这个代码把单个模型产生的结果合并一下
'''
'''
因为分别对5个类进行训练，所以出一个类的模型就把test.csv相关类的结果进行替换代码如下:
'''
# PART_IMAGE_CATEGORY='outwear'                      #类别
# RESUTL_PART_PATH='../data_b/tem_resutl/outwear_result_b0024.csv'  #单类结果
# RESULT_ALL_PATH='../submit/test.csv'    #所有类别的结果
#
# result_part=pd.read_csv(RESUTL_PART_PATH)
# print("signal category counts:",result_part.shape[0])
#
# result_all=pd.read_csv(RESULT_ALL_PATH)
# result_all_count=result_all.shape[0]
# print("all category counts:",result_all_count)
#
# result_all=result_all[result_all['image_category']!=PART_IMAGE_CATEGORY]
# result=pd.concat([result_part,result_all])
# result_all_after_concat=result.shape[0]
# print("all catgory counts after concat: ",result_all_after_concat)
#
# assert result_all_count==result_all_after_concat #做个断言检查测试结果的个数是否和测试图片的个数相同。
# result.to_csv(RESULT_ALL_PATH,index=False)
# os.system('zip -rj ../submit/test.zip ../submit/test.csv')#shen
# print('finshed !')
'''
下面这段代码是同时对5个结果合并
'''
blouse_result=pd.read_csv('../submit/blouse_result_b.csv')
dress_resutl=pd.read_csv('../submit/dress_result_b.csv')
outwear_resutl=pd.read_csv('../submit/outwear_result_b.csv')
skirt_resutl=pd.read_csv('../submit/skirt_result_b.csv')
trousers_resutl=pd.read_csv('../submit/trousers_result_b.csv')
result=pd.concat([blouse_result,dress_resutl,outwear_resutl,skirt_resutl,trousers_resutl])
print(result.shape[0])
test=pd.read_csv('../submit/test.csv')
print(test.shape[0])

assert result.shape[0] == test.shape[0]#做个断言检查测试结果的个数是否和测试图片的个数相同。
result.to_csv('../submit/test.csv' ,index=False)

