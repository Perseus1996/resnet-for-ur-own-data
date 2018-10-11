# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import pandas as pd
import os
from PIL import Image
import makedata
model_path=r'./train_image_63.model-200' 
test_path='data\\guangdong_round1_test_a_20180916'
x=tf.placeholder(tf.float32,[None,256,256,3])     
pred,endpoints=nets.resnet_v2.resnet_v2_50(x,num_classes=12,is_training=True,reuse=True)
pred=tf.reshape(pred,[-1,12])
result=tf.argmax(pred,1)
saver=tf.train.Saver()
test_classes=makedata.classes
image_name=[]
dect_label=[]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,model_path)
    for path in os.listdir(test_path):
        image_name.append(path)
        path=os.path.join(test_path,path)
        image=Image.open(path)
        image=image.resize((256,256))
        image=tf.reshape(image,[256,256,3])
        image=tf.image.per_image_standardization(image)
        image_test=tf.reshape(image,[256,256,3])
        b_image=sess.run([image_test])
        pred_value=sess.run(result,feed_dict={x:b_image})
        index=pred_value[0]
        dect_label.append(test_classes[index])
    label_file = pd.DataFrame({'img_name': image_name, 'dect_lable': dect_label})
    label_file.to_csv('data//result.csv', index=False)
            
        
        
        
        
        
        
        