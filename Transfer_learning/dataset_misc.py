"""Functions for downloading and reading data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import PIL.Image as Image
import random
import numpy as np
import cv2
import time
import math
import lscript as lsp

def get_frames_data(filename, num_frames_per_clip=16):
  ''' Given a directory containing extracted frames, return a video clip of
  (num_frames_per_clip) consecutive frames as a list of np arrays '''
  ret_arr = []
  s_index = 0
  for parent, dirnames, filenames in os.walk(filename):
    if(len(filenames)<num_frames_per_clip):
      return [], s_index
    filenames = sorted(filenames)
    s_index = random.randint(0, len(filenames) - num_frames_per_clip)
    for i in range(s_index, s_index + num_frames_per_clip):
      image_name = str(filename) + '/' + str(filenames[i])
      img = Image.open(image_name)
      img_data = np.array(img)
      #lsp.view_image(img_data)
      ret_arr.append(img_data)
  print('ret_arr',np.array(ret_arr).shape)
  return ret_arr, s_index

def get_compress_frames_data(filename, num_frames_per_clip=16):
  ''' Given a directory containing extracted frames, return a video clip of
  (num_frames_per_clip) consecutive frames as a list of np arrays '''
  ret_arr = []
  s_index = 0
  for parent, dirnames, filenames in os.walk(filename):
    if(len(filenames)<num_frames_per_clip):
      return [], s_index
    filenames = sorted(filenames)
    s_index = random.randint(0, len(filenames) - num_frames_per_clip)
    jump=math.floor((len(filenames)/num_frames_per_clip))
    #print('jump',jump,'toatl lenght of file',len(filenames))
    loop=0
    for i in range(0,len(filenames),jump):
      #print('loop',i)
      #print(i)
      if (loop>15):
        break
      image_name = str(filename) + '/' + str(filenames[i])
      img = Image.open(image_name)
      img_data = np.array(img)
      #lsp.view_image(img_data)
      ret_arr.append(img_data)
      loop=loop+1
  #print('ret_arr',np.array(ret_arr).shape)
  return ret_arr, s_index


def read_vid_and_label(filename, batch_size, start_pos=-1, num_frames_per_clip=16, crop_size=112,normalisation=True):
  lines = open(filename,'r')
  read_dirnames = []
  data = []
  label = []
  batch_index = 0
  next_batch_start = -1
  lines = list(lines)
  # Forcing shuffle, if start_pos is not specified
  if start_pos < 0:
    video_indices = list(range(len(lines)))
    random.seed(time.time())
    random.shuffle(video_indices)
    #print(video_indices)
  else:
    # Process videos sequentially
    video_indices = range(start_pos, len(lines))
  iter=0
  for index in video_indices:
    #print(batch_index)
    #print(index)
    if(batch_index>=batch_size):
      next_batch_start = index
      break
    line = lines[index].strip('\n').split()
    dirname = line[0]
    tmp_label = line[1]
    print("Loading {}...".format(dirname))
    tmp_data, _ = get_compress_frames_data(dirname, num_frames_per_clip)
    #print('temp_data.shape',np.array(tmp_data))
    img_datas = []
    if(len(tmp_data)!=0):
      for j in xrange(len(tmp_data)):
        img = np.array(tmp_data[j].astype(np.uint8))
        img = cv2.resize(img,(crop_size,crop_size))
        #lsp.view_image(img)
        img_datas.append(img)
        #lsp.view_image(img_datas[0])
      data.append(img_datas)
      #lsp.view_image(data[iter][0])
      iter=iter+1
      label.append(int(tmp_label))
      batch_index = batch_index + 1
      read_dirnames.append(dirname)

  #lsp.view_video_seq(data,label,16,1) 

  np_arr_data = np.array(data) #.astype(np.float32)
  print(np_arr_data.shape)
  np_arr_label = np.array(label) #.astype(np.int64)
  
  if normalisation:
    np_arr_data=np_arr_data/255
    
  #lsp.view_image(np_arr_data[0][0])
  #lsp.view_video_seq(np_arr_data,np_arr_label,16,1) 

  return np_arr_data, np_arr_label, next_batch_start, read_dirnames



def read_vid_and_label_rand_frames(filename, batch_size, start_pos=-1, num_frames_per_clip=16, crop_size=112,normalisation=True):
  lines = open(filename,'r')
  read_dirnames = []
  data = []
  label = []
  batch_index = 0
  next_batch_start = -1
  lines = list(lines)
  # Forcing shuffle, if start_pos is not specified
  if start_pos < 0:
    video_indices = list(range(len(lines)))
    random.seed(time.time())
    random.shuffle(video_indices)
    #print(video_indices)
  else:
    # Process videos sequentially
    video_indices = range(start_pos, len(lines))
  iter=0
  for index in video_indices:
    #print(batch_index)
    #print(index)
    if(batch_index>=batch_size):
      next_batch_start = index
      break
    line = lines[index].strip('\n').split()
    dirname = line[0]
    tmp_label = line[1]
    print("Loading {}...".format(dirname))
    tmp_data, _ = get_frames_data(dirname, num_frames_per_clip)
    #print('temp_data.shape',np.array(tmp_data))
    img_datas = []
    if(len(tmp_data)!=0):
      for j in xrange(len(tmp_data)):
        img = np.array(tmp_data[j].astype(np.uint8))
        img = cv2.resize(img,(crop_size,crop_size))
        #lsp.view_image(img)
        img_datas.append(img)
        #lsp.view_image(img_datas[0])
      data.append(img_datas)
      #lsp.view_image(data[iter][0])
      iter=iter+1
      label.append(int(tmp_label))
      batch_index = batch_index + 1
      read_dirnames.append(dirname)

  #lsp.view_video_seq(data,label,16,1) 

  np_arr_data = np.array(data) #.astype(np.float32)
  print(np_arr_data.shape)
  np_arr_label = np.array(label) #.astype(np.int64)
  
  if normalisation:
    np_arr_data=np_arr_data/255
    
  #lsp.view_image(np_arr_data[0][0])
  #lsp.view_video_seq(np_arr_data,np_arr_label,16,1) 

  return np_arr_data, np_arr_label, next_batch_start, read_dirnames







def read_clip_and_label(filename, batch_size, start_pos=-1, num_frames_per_clip=16, crop_size=112, shuffle=False):
  ''' Original function'''
  lines = open(filename,'r')
  read_dirnames = []
  data = []
  label = []
  batch_index = 0
  next_batch_start = -1
  lines = list(lines)
  # Forcing shuffle, if start_pos is not specified
  if start_pos < 0:
    shuffle = True
  if shuffle:
    video_indices = range(len(lines))
    random.seed(time.time())
    random.shuffle(list(video_indices))
  else:
    # Process videos sequentially
    video_indices = range(start_pos, len(lines))
  for index in video_indices:
    if(batch_index>=batch_size):
      next_batch_start = index
      break
    line = lines[index].strip('\n').split()
    dirname = line[0]
    tmp_label = line[1]
    if not shuffle:
      print("Loading a video clip from {}...".format(dirname))
    tmp_data, _ = get_frames_data(dirname, num_frames_per_clip)
    img_datas = []
    if(len(tmp_data)!=0):
      for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if(img.width>img.height):
          scale = float(crop_size)/float(img.height)
          img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), crop_size))).astype(np.float32)
        else:
          scale = float(crop_size)/float(img.width)
          img = np.array(cv2.resize(np.array(img),(crop_size, int(img.height * scale + 1)))).astype(np.float32)
        img = img[int((img.shape[0] - crop_size)/2):int((img.shape[0] - crop_size)/2) + crop_size, int((img.shape[1] - crop_size)/2):int((img.shape[1] - crop_size)/2) + crop_size,:]
        img_datas.append(img)
      data.append(img_datas)
      label.append(int(tmp_label))
      batch_index = batch_index + 1
      read_dirnames.append(dirname)

  # pad (duplicate) data/label if less than batch_size
  valid_len = len(data)
  pad_len = batch_size - valid_len
  if pad_len:
    for i in range(pad_len):
      data.append(img_datas)
      label.append(int(tmp_label))

  np_arr_data = np.array(data) #.astype(np.float32)
  np_arr_label = np.array(label).astype(np.int64)

  return np_arr_data, np_arr_label, next_batch_start, read_dirnames, valid_len



def check():
    print('Checked')
   