'''
Helper functions for:
Items {
1:'viewing single image'
2:'viewing multiple images'
3:''}

Author: @leopauly
'''

# setting seeds
from numpy.random import seed
seed(1)
import os
os.environ['PYTHONHASHSEED'] = '2'
import tensorflow as tf
tf.set_random_seed(3)

#Imports
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

def check():
    print('lscript checked')

def view_image(img,image_details=False):
    '''Method for displaying a single image'''
    if (image_details):
        print('shape of image: {}, Type of image {}: '.format(np.shape(img),img.dtype))
        print('Image array \n:',img)
    #plt.figure(figsize=(20,20))
    plt.imshow(img)
    plt.gray()
    plt.show()
    
def view_images(img,labels,axis_show='off'):
    ''' Displaying multiple images as subplots '''
    plt.figure(figsize=(20,20))#figsize=(16,16))
    for i,_ in enumerate(img):
            plt.subplot(3,20,i+1)
            plt.imshow(img[i])
            plt.axis(axis_show)
            plt.title(str(labels[i]))
    plt.gray()
    plt.show()
    
def reshape_grayscale_as_tensor(batch_x):
    ''' reshape numpy grayscale image arrays into tensor format'''
    batch_x = batch_x.reshape(batch_x.shape[0],batch_x.shape[1], batch_x.shape[2],1)
    return batch_x

def reshape_rgb_as_tensor(batch_x):
    ''' reshape numpy grayscale image arrays into tensor format'''
    batch_x = batch_x.reshape(batch_x.shape[0],batch_x.shape[1], batch_x.shape[2],3)
    return batch_x

def plot_values_with_legends(x,y,legend_to_plot,x_axis,y_axis,title,color='red',ylim=True):
    patch = mpatches.Patch(color=color, label=legend_to_plot)
    plt.figure(figsize=(20,5))
    plt.plot(x,y)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid(True)
    #plt.ylim((-.2,0))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(title)
    plt.legend(handles=[patch])
    plt.show()

def view_video_seq(x,y,time_step,item_num):
    print('label:{}'.format(y[item_num]))
    print('Video_seq shape:',x.shape,'Label shape',y.shape)
    for i in range (0,time_step):
        img=x[item_num][i]
        view_image(img)
        
def view_video_inline(x,y,time_step,item_num,axis_show='off'):
    
    import numpy as np
    import matplotlib.pyplot as plt
 
    plt.figure(figsize=(30,30))
    for i in range (0,time_step):
        img=x[item_num][i]
        plt.subplot(1,time_step,i+1)
        plt.imshow(img,cmap='viridis')
        plt.axis(axis_show)
    #plt.color()
    plt.show()  
    
        
def one_hot(y,nb_classes):
    return np_utils.to_categorical(y,nb_classes)
        
def leo():
    print('leo {}'.format(3))
    
