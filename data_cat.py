import pandas as pd
import numpy as np
import os
'''
因为分开训练了5个模型，所以用这个代码把单个模型产生的结果合并一下
'''

PART_IMAGE_CATEGORY='dress'                      #类别
RESUTL_PART_PATH='./data/test/dress_result0417_1.csv'  #单类结果
RESULT_ALL_PATH='./data/test/test.csv'    #所有类别的结果

result_part=pd.read_csv(RESUTL_PART_PATH)
print("signal category counts:",result_part.shape[0])

result_all=pd.read_csv(RESULT_ALL_PATH)
result_all_count=result_all.shape[0]
print("all category counts:",result_all_count)

result_all=result_all[result_all['image_category']!=PART_IMAGE_CATEGORY]
result=pd.concat([result_part,result_all])
result_all_after_concat=result.shape[0]
print("all catgory counts after concat: ",result_all_after_concat)

assert result_all_count==result_all_after_concat
result.to_csv(RESULT_ALL_PATH,index=False)
os.system('zip -rj ./data/test/test.zip ./data/test/test.csv')#生成test.zip以供提交

print('finshed !')
