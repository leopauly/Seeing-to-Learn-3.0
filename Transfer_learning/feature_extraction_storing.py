

## Description
print('Author : @leopauly')
print('Program: Feature extraction and storing')
print('Change:')
print('1. Model folder')
print('2. Model graph')
print('3. Iter number')
print('4. Load weights')
print('5. Layer name')
print('6. storage file name')

#-----------------------------------------------------------------------------------------------------------------------#

## Imports
import os
from six.moves import xrange 
import PIL.Image as Image
import random
import numpy as np
import cv2
import time
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
import h5py

# Custom scripts
import lscript as lsp
import modelling as md

#-----------------------------------------------------------------------------------------------------------------------#



height=112 
width=112 
channel=3
crop_size=112
cluster_length=16
feature_size=8192
num_videos_per_class_test=100000
num_class_test=5
load_saved_weights=True
saved_path='/nobackup/leopauly/S2l/'

#-----------------------------------------------------------------------------------------------------------------------#


## Loading pre-trained model
x_image = tf.placeholder(tf.float32, [None, cluster_length,height,width,channel],name='x') 
model_keras = md.C3D_ucf101_training_model_tf(summary=True)
out=model_keras(x_image)
print('Miscellenious items finished..!!',flush=True)


## Start the session with logging placement.
init_op = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
sess.run(init_op)

## Restore model weights from previously saved model
saver = tf.train.Saver()
if(load_saved_weights):
    saver.restore(sess, os.path.join(saved_path,'activity_model.ckpt-67'))
    print("Model restored from file: %s" % saved_path,flush=True)


#-----------------------------------------------------------------------------------------------------------------------#
    
    
    
## Feature extraction from testing data


## Extraction of features 
def extract_video_features(vid):
    vid_=vid.reshape(-1,cluster_length,height,width,channel)
    f_v = sess.graph.get_tensor_by_name('pool4/MaxPool3D:0') 
    #('flatten_1/Reshape:0') #('pool4/MaxPool3D:0') #('dropout_2/cond/Merge:0') #('fc8/BiasAdd:0') 
    f_v_val=sess.run([f_v], feed_dict={'conv1_input:0':vid_,x_image:vid_,K.learning_phase(): 0 })
    features=np.reshape(f_v_val,(-1))
    return features


## Uniform smapling of frames from the entire video
def get_compress_frames_data(video_filepath, num_frames_per_clip=16):
  ret_arr = []
  for _, _, filenames in os.walk(video_filepath):
    filenames = sorted(filenames)
    jump=math.floor((len(filenames)/num_frames_per_clip))
    loop=0
    for i in range(0,len(filenames),jump):
      if (loop>15): break
      if (filenames[i].endswith('.png')):
        image_name = str(video_filepath) + '/' + str(filenames[i])
        img = Image.open(image_name)
        img_data = np.array(img)
        img_data = cv2.resize(img_data,(crop_size,crop_size))
        ret_arr.append(img_data)
        loop=loop+1
  ret_arr=np.array(ret_arr) #ret_arr=ret_arr/255
  return ret_arr


## Feature extraction
def get_features_from_class(class_folder):
    feature_set_a=[]
    base_dir_a=class_folder
    sub_dir_a=os.listdir(base_dir_a)
    sub_dir_a=sorted(sub_dir_a)
    if '.DS_Store' in sub_dir_a:
        sub_dir_a.remove('.DS_Store')
    num_videos_per_class_test_loop=0
    for sub_dir_a_ in sub_dir_a:
        sub_sub_dir_a=os.listdir(base_dir_a+sub_dir_a_+'/')
        for sub_sub_dir_a_ in sub_sub_dir_a:
            if(num_videos_per_class_test_loop>=num_videos_per_class_test): break
            video_filepath=base_dir_a+sub_dir_a_+'/'+sub_sub_dir_a_
            print('Extracting feature from:',video_filepath)
            vid_a=get_compress_frames_data(video_filepath)
            feature_set_a.append(extract_video_features(vid_a))
            num_videos_per_class_test_loop+=1
    return np.array(feature_set_a)

#-----------------------------------------------------------------------------------------------------------------------#

## Feature extraction
feature_set_a=get_features_from_class('/nobackup/leopauly/MIME/MIME_videos_frames/1/')
feature_set_b=get_features_from_class('/nobackup/leopauly/MIME/MIME_videos_frames/2/')
feature_set_c=get_features_from_class('/nobackup/leopauly/MIME/MIME_videos_frames/3/')
feature_set_d=get_features_from_class('/nobackup/leopauly/MIME/MIME_videos_frames/4/')
feature_set_e=get_features_from_class('/nobackup/leopauly/MIME/MIME_videos_frames/5/')

#-----------------------------------------------------------------------------------------------------------------------#



## Storing test data: Features
storage_filename='/nobackup/leopauly/S2l3.0/Features_Storage/feature_trainUCF101_testmime5_pool4.h5'
os.system('rm %s'%storage_filename)
hf = h5py.File(storage_filename, 'w')
hf.create_dataset('feature_set_a', data=feature_set_a)
hf.create_dataset('feature_set_b', data=feature_set_b)
hf.create_dataset('feature_set_c', data=feature_set_c)
hf.create_dataset('feature_set_d', data=feature_set_d)
hf.create_dataset('feature_set_e', data=feature_set_e)
hf.close()