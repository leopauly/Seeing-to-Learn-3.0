'''
Helper functions for preparing datasets
Author : @leopauly

'''
#Imports
import numpy as np
import keras 
from random import randint

def batch_gen(batch_start,batch_stop,batch_size,time_step,h,w,ch,imagefolderpath,gray,normalisation=True):
    ''' Genreates batches of video sequences suitable for feeding to a ConvLSTM
    '''
    import cv2
    
    if (ch==1):
        X=np.zeros([batch_size,time_step,h,w])
    else:
        X=np.zeros([batch_size,time_step,h,w,ch])
    for i in range(batch_start,batch_stop):
        j_new=0
        for j in range(0+(50*i),50+(50*i)):
            img=cv2.imread(str(imagefolderpath+str(j)+'.png'))
            if (gray==True):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if(normalisation==True):
                img = img.astype('float32')
                img=img/255
            X[i][j_new]=np.array(img)
            j_new=j_new+1
    return X


def batch_gen_train(nb_classes,batch_size,time_step,h,w,ch,imagefolderpath,gray,normalisation=True):
    ''' Genreates batches of video sequences suitable for feeding to a ConvLSTM
        Remember : Change definition for seq
    '''
    import cv2
    
    if (ch==1):
        X=np.zeros([batch_size,time_step,h,w])
    else:
        X=np.zeros([batch_size,time_step,h,w,ch])
    y=np.zeros([batch_size])
    
    for i in range(batch_size):
        j_new=0
        action_id=randint(0,nb_classes-1)
        seq=(randint(0,110)*10)
        for j in range(seq,seq+time_step):
           
            if (gray==True):
                img = np.array(cv2.imread(str(imagefolderpath+'/'+str(action_id)+'/'+str(j)+'.png'),0))
            else:
                img = np.array(cv2.imread(str(imagefolderpath+'/'+str(action_id)+'/'+str(j)+'.png')))   
            if(normalisation==True):
                img = img.astype('float32')
                img=img/255
    
            X[i][j_new]=np.array(img)
            j_new=j_new+1
        y[i]=action_id   
    return X,y


def batch_gen_test(nb_classes,batch_size,time_step,h,w,ch,imagefolderpath,gray,normalisation=True):
    ''' Genreates batches of video sequences suitable for feeding to a ConvLSTM
        Remember : Change definition for seq
    '''
    
    import cv2
    
    if (ch==1):
        X=np.zeros([batch_size,time_step,h,w])
    else:
        X=np.zeros([batch_size,time_step,h,w,ch])
    y=np.zeros([batch_size])
    
    for i in range(batch_size):
        j_new=0
        action_id=randint(0,nb_classes-1)
        seq=(randint(110,120)*10)
        for j in range(seq,seq+50):
           
            if (gray==True):
                img = np.array(cv2.imread(str(imagefolderpath+'/'+str(action_id)+'/'+str(j)+'.png'),0))
            else:
                img = np.array(cv2.imread(str(imagefolderpath+'/'+str(action_id)+'/'+str(j)+'.png')))   
            if(normalisation==True):
                img = img.astype('float32')
                img=img/255
            
            X[i][j_new]=np.array(img)
            j_new=j_new+1
        y[i]=action_id   
    return X,y

def cluster_generator(x,num_clusters,cluster_length,height,width,channel):
    clusters=np.zeros([num_clusters,cluster_length,height,width,channel])
    print(clusters.shape)
    for i in range(num_clusters):
        w=x[0,(i*cluster_length):((i+1)*cluster_length)]
        w=w.reshape(1,w.shape[0],height,width,channel)
        clusters[i]=w
    return clusters

def clusters_with_strides(x,num_clusters,cluster_length,height,width,channel,strides):
    clusters=np.empty([num_clusters,cluster_length,height,width,channel])
    print(clusters.shape)
    for i in range(num_clusters):
        w=x[0,(i*strides):((i*strides)+cluster_length)]
        w=w.reshape(1,w.shape[0],height,width,channel)
        clusters[i]=w
    return clusters