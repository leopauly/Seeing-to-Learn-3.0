## Training C3D network with MIME dataset
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
import mime20_dataset as mime
print('Loaded libraries...!!',flush=True)

height=112 
width=112 
channel=3
cluster_length=16
nb_classes=20
lr_rate=.001
next_batch_start=0
batch_size=16
batch_size_test=16
total_train_videos=30360
memory_batch_size_train=30360
memory_batch_size_test=3360
iterations= 30 
custom_global_step=0
LOG_DIR='/nobackup/leopauly/S2l/MIME/90_10_shuffle'
#saved_path='/nobackup/'
best_validation_accuracy=0.0
best_iteration=0
print('Finished defining variables..!!',flush=True)

#------------------------------------------------------------------------------------------------------------------------#


## Finding how many devices are available
gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
num_gpus = len(gpus)
print("GPU nodes found: " + str(num_gpus),flush=True)
for i in range(num_gpus):
    print('Avaialble gpu:',str(gpus[i]),flush=True)
    
    
## Finding how many CPUs are available
cpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU']
num_cpus = len(cpus)
print("CPU nodes found: " + str(num_cpus),flush=True)
for i in range(num_cpus):
    print('Avaialble cpu:',str(cpus[i]),flush=True)


## Defining placeholders in tf for images and targets
x_image = tf.placeholder(tf.float32, [None, 16,height,width,channel]) 
y_true = tf.placeholder(tf.float32, [None, nb_classes])
y_true_cls = tf.placeholder(tf.int64, [None])

print('Finished obtaining gpu details and place holders',flush=True)


#------------------------------------------------------------------------------------------------------------------------#

## Creating the model and defining the function to parallelise data
## Define the network in a model function, to make parallelisation across GPUs easier.
def model(x_image_, y_true_):
    ''' Expecting the following parameters, in batches:
        x_image_ - x_image batch
        y_true_ - y_true batch
    '''

    model_keras = md.C3D_MIME20_training_model_tf(summary=False)
    
    out=model_keras(x_image_)
    print(out,flush=True)
    
    y_pred = tf.nn.softmax(out)
    y_pred_cls = tf.argmax(out, dimension=1)
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true_,logits=out))
    
    # Outputs to be returned to CPU
    return y_pred, y_pred_cls, loss


def make_parallel(fn, **kwargs):
    in_splits = {}
    for k, v in kwargs.items():
        in_splits[k] = tf.split(v, num_gpus)
    # An array for every aggregated output
    y_pred_split, y_pred_cls_split, cost_split = [], [], []
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=i > 0):
                y_pred_, y_pred_cls_, cost_,  = fn(**{k : v[i] for k, v in in_splits.items()})
                # Adding the output from each device.
                y_pred_split.append(y_pred_)
                y_pred_cls_split.append(y_pred_cls_)
                cost_split.append(cost_)
    return tf.concat(y_pred_split, axis=0), tf.concat(y_pred_cls_split, axis=0),tf.stack(cost_split, axis=0)

print('Finished defining special functions',flush=True)
  
if num_gpus > 0:
    # There is significant latency for CPU<->GPU copying of shared variables.
    # We want the best balance between speedup and minimal latency.
    y_pred, y_pred_cls, cost = make_parallel(model, x_image_=x_image, y_true_=y_true)
else:
    # CPU-only version
    y_pred, y_pred_cls, cost = model(x_image_=x_image, y_true_=y_true)


# Optimisation calculated on CPU on aggregated results.
# NEED the colocate_gradients_with_ops flag TRUE to get the gradient ops to run on same device as original op!
optimizer = tf.train.AdagradOptimizer(learning_rate=2e-4).minimize(cost, colocate_gradients_with_ops=True)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Miscellenious items finished..!!',flush=True)

#------------------------------------------------------------------------------------------------------------------------#

## Training & testing
def testing(iterations,loops):
    test_score=0
    for j in range(int(memory_batch_size_test/batch_size_test)-1):
        test_score_ = sess.run([accuracy], feed_dict={x_image:test_images[(batch_size_test*j):(batch_size_test*(j+1))],y_true_cls:test_labels_cls[(batch_size_test*j):(batch_size_test*(j+1))],K.learning_phase(): 0 })
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
train_images, train_labels_cls, next_batch_start, _ = mime. read_vid_and_label('./train_90_10_shuffle.list',memory_batch_size_train,-1,cluster_length,112,normalisation=False)
end_time = time.time()
time_dif = end_time - start_time
print("Time usage for loading training dataset: " + str(timedelta(seconds=int(round(time_dif)))),flush=True)
train_labels=keras.utils.to_categorical(train_labels_cls, num_classes=nb_classes)

## Loading Testing data
test_images, test_labels_cls, next_batch_start, _ = mime.read_vid_and_label('./test_90_10_shuffle.list',memory_batch_size_test,-1,16,112,normalisation=False)
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

