## Training Encoder network with UCF101 and contrstive loss
## Author : @leopauly | www.leopauly.com
print('Started running the program..!',flush=True)

#------------------------------------------------------------------------------------------------------------------------#

## Imports
from keras.models import Sequential
import random
import numpy as np
from PIL import Image
from os import listdir
from scipy.ndimage import imread
import keras
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras import backend as K
import datetime
import time
import os 
from datetime import timedelta
## Custom scripts
import lscript as lsp
import modelling as md
import dataset as dset
import dataset_misc as dset_misc
print('Loaded libraries...!!',flush=True)


height=112 
width=112 
channel=3
cluster_length=16
lr_rate=.001
next_batch_start=0
batch_size=16
nb_classes=20
batch_size_test=16
memory_batch_size_train=200#30360
memory_batch_size_test=200#3360
iterations= 30 
custom_global_step=0
LOG_DIR='/nobackup/leopauly/S2l3.0/Trained_Models/UCF101_contrastive_loss/'
best_validation_accuracy=0.0
best_iteration=0
print('Finished defining variables..!!',flush=True)

#------------------------------------------------------------------------------------------------------------------------#

## Defining placeholders in tf for images and targets
x_image = tf.placeholder(tf.float32, [None, 16,height,width,channel]) 
y_true = tf.placeholder(tf.float32, [None, nb_classes])
print('Finished obtaining gpu details and place holders',flush=True)

#------------------------------------------------------------------------------------------------------------------------#

## Contrastive loss
def contrstive_loss(left_output,right_output,label):
    margin = 0.2
    d = tf.reduce_sum(tf.square(left_output - right_output), 1)
    d_sqrt = tf.sqrt(d)
    loss = label * tf.square(tf.maximum(0., margin - d_sqrt)) + (1 - label) * d
    loss = 0.5 * tf.reduce_mean(loss)
    return loss

## Define the network in a model function
def model(x_image_, y_true_):
    
    model_keras = md.encoder_training_model_tf(summary=False)
    out=model_keras(x_image_)
    print(out,flush=True)
    
    y_pred = tf.linalg.norm(a - b, axis=1)
    loss=y_true_ * tf.square(out) + (1.0 - y_true_) * tf.square(tf.maximum(margin - out, 0.0))
    return loss
    
    
    for i,left_output in enumerate(out):
        for j,right_output in enumerate(out): 
            if (y_true_[i]==y_true_[j]):
                contrast_label=0
            else:
                contrast_label=1
            loss=loss+contrstive_loss(left_output,right_output,contrast_label)  
    return loss


## Optimisation
cost = model(x_image_=x_image, y_true_=y_true)
optimizer = tf.train.AdagradOptimizer(learning_rate=2e-4).minimize(cost, colocate_gradients_with_ops=True)
print('Miscellenious items finished..!!',flush=True)

#------------------------------------------------------------------------------------------------------------------------#

## Training & testing
def testing(iterations,loops):
    test_score=0
    for j in range(int(memory_batch_size_test/batch_size_test)-1):
        test_score_ = sess.run([cost], feed_dict={x_image:test_images[(batch_size_test*j):(batch_size_test*(j+1))],y_true_cls:test_labels_cls[(batch_size_test*j):(batch_size_test*(j+1))],K.learning_phase(): 0 })
        test_score=test_score+sum(test_score_)
    print('Test score after iteration:',iterations,',loop:',loops,'is:',test_score/(j+1),flush=True)
    validation_accuracy= (test_score/(j+1))*100
    return validation_accuracy


#------------------------------------------------------------------------------------------------------------------------#

## Start the session with logging placement.
init_op = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
sess.run(init_op)

## Restore model weights from previously saved model 
saver = tf.train.Saver()
#saver.restore(sess, os.path.join(saved_path,'activity_model.ckpt-67'))
#print("Model restored from file: %s" % saved_path,flush=True)
#print ('started the session...!!',flush=True)

## Loading Training data
start_time = time.time()
train_images, train_labels_cls, next_batch_start, _ = dset_misc. read_vid_and_label('./UCF101_data_preparation/train.list',memory_batch_size_train,-1,cluster_length,112,normalisation=False)
end_time = time.time()
time_dif = end_time - start_time
print("Time usage for loading training dataset: " + str(timedelta(seconds=int(round(time_dif)))),flush=True)
train_labels=keras.utils.to_categorical(train_labels_cls, num_classes=nb_classes)

## Loading Testing data
test_images, test_labels_cls, next_batch_start, _ = dset_misc.read_vid_and_label('./UCF101_data_preparation/test.list',memory_batch_size_test,-1,16,112,normalisation=False)
test_labels=keras.utils.to_categorical(test_labels_cls, num_classes=nb_classes)
print('testing data loaded',flush=True)


#------------------------------------------------------------------------------------------------------------------------#

## Training
for i in range(0,(iterations*10)):
    print('started iteration:',i,flush=True)
    for j in range (int(memory_batch_size_train/batch_size)-1):    
        print ('This is epoch:',j,'going to be trained',flush=True)
        output_value = sess.run([optimizer], feed_dict={x_image:train_images[(batch_size*j):(batch_size*(j+1))],y_true:train_labels[(batch_size*j):(batch_size*(j+1))],K.learning_phase(): 1 })   
        print ('This is epoch:',j,'trained',flush=True)     
    validation_accuracy=testing(i,j)
    print(' validation_accuracy in iteration:',i,'is:', validation_accuracy)
    
    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        best_iteration=i
        saver.save(sess, os.path.join(LOG_DIR, "activity_model.ckpt"), global_step=i)
        custom_global_step=custom_global_step+1   
        print('Model saved after iteration:',best_iteration,flush=True)

print('best_validation_accuracy:',best_validation_accuracy)      
print('best_iteration:',best_iteration)

sess.close()


#------------------------------------------------------------------------------------------------------------------------#

