# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import readdata
save_dir=r'./train_image_63.model'
batch_size=2
lr=tf.Variable(0.0001,tf.float32)
x=tf.placeholder(tf.float32,[None,256,256,3])
y_=tf.placeholder(tf.float32,[None])
image_batch,lable_batch=readdata.get_tfrecord(batch_size,isTrain=True)
one_hot_lables=tf.one_hot(indices=tf.cast(y_,tf.int32),depth=12)
pred,endpoint=nets.resnet_v2.resnet_v2_50(x,num_classes=12,is_training=True)
pred=tf.reshape(pred,shape=[-1,12])
loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=one_hot_lables))
optimizer=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
a=tf.argmax(pred,1)
b=tf.argmax(one_hot_lables,1)
correct_pred=tf.equal(a,b)
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    threads=tf.train.start_queue_runners(sess=sess)
    i=0
    while True:
        i+=1
        print(i)
        b_image,b_lable=sess.run([image_batch,lable_batch])
        _,loss_,y_t,y_p,a_,b_=sess.run([optimizer,loss,one_hot_lables,pred,a,b],
        feed_dict={x:b_image,y_:b_lable})
        print(loss_)
        if i==1500:
            saver.save(sess,save_dir,global_step=i)
            break

        