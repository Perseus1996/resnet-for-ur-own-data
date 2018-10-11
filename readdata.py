# -*- coding: utf-8 -*-
import os
import tensorflow as tf 
from PIL import Image
tf_record_train_path='data\\train_data.tfrecord'
def read_tfrecord(path):
    filename_queue=tf.train.string_input_producer([path])
    reader=tf.TFRecordReader()
    _,serialized_example=reader.read(filename_queue)
    features=tf.parse_single_example(serialized_example,features={
    'lable':tf.FixedLenFeature([],tf.int64),
    'image_raw':tf.FixedLenFeature([],tf.string)})
    image=tf.decode_raw(features['image_raw'],tf.uint8)
    image=tf.reshape(image,[256,256,3])
    image=tf.image.per_image_standardization(image)
    lable=tf.cast(features['lable'],tf.int32)
    return image,lable
def get_tfrecord(batch_size,isTrain=True):
    if isTrain:
        tf_record_path=tf_record_train_path
#    else:
#        tf_record_path=data_test_path
    image,lable=read_tfrecord(tf_record_path)
    image_batch,lable_batch=tf.train.shuffle_batch([image,lable],batch_size=batch_size,
    num_threads=1,capacity=1000,min_after_dequeue=10)
    return image_batch,lable_batch
    
