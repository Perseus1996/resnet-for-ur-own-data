# -*- coding: utf-8 -*-
"""
@author: perseus
"""
import os
import tensorflow as tf 
from PIL import Image
import numpy as np
data_path= 'data\\guangdong_round1_train2_20180916'#也可以输入你自己的test路径
writer=tf.python_io.TFRecordWriter('data\\train_data.tfrecord')
classes=['正常',
        '不导电',
        '擦花',
        '横条压凹',
        '桔皮',
        '漏底',
        '碰伤',
        '起坑',
        '凸粉',
        '涂层开裂',
        '脏点',
        '其他'
        ]#也可以输入你自己的分类
class_path=[]
for index,name in enumerate(classes):
    if name=='正常':
        class_path.append(os.path.join(data_path,'无瑕疵样本'))
    else:
        if name=='其他':
            new_path=os.path.join(data_path,'瑕疵样本','其他')
            for newpath in os.listdir(new_path):
                class_path.append(os.path.join(data_path,'瑕疵样本',name,newpath)) 
        else:
            class_path.append(os.path.join(data_path,'瑕疵样本',name))
for path in class_path:
    for image_path in os.listdir(path):
        image_path=os.path.join(path,image_path)
        image= Image.open(image_path)
#        image=np.asarray(image,np.uint8)
        image=image.resize((256,256))
        image_raw=image.tobytes()
        example=tf.train.Example(features=tf.train.Features(feature={
    'lable':tf.train.Feature(int64_list=tf.train.Int64List(value=[index])) ,
    'image_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))                                                           
    }))
    writer.write(example.SerializeToString())
writer.close() 
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    