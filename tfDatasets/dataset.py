# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import labels_diz as lbl
import numpy as np
import helper

import matplotlib.pyplot as plt


labels_diz = lbl.labels_diz
num_classes = len(labels_diz)

alfa = (127,) #semi-transparent
colors = np.array([label.color + alfa for label in labels_diz]).astype(np.uint8)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

load_net = '/media/undead/Data/Segmentation/net/my2-net-73949'

saver = tf.train.import_meta_graph(load_net + '.meta')
saver.restore(sess,load_net)


model = tf.get_default_graph()

input_image = model.get_tensor_by_name('image_input:0')
keep_prob = model.get_tensor_by_name('keep_prob:0')
nn_output = model.get_tensor_by_name('layer3_up/BiasAdd:0')

logits = tf.reshape(nn_output,(-1,num_classes))


data_folder='/media/undead/8Tb/out'
image_shape=(576,1024)
dataname = 'data/'

ride = 'test13_4'
camera = 'argus_cam_5'

data_folder = os.path.join(data_folder,ride, camera)
out_folder = os.path.join('/media/undead/ssd/Voxels/',ride,camera)
l = glob(os.path.join(data_folder, dataname, '*.jpg'))


road_name = 'Xroad'
overlay_name = 'Xoverlay'
#%%
def _parse_function(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, image_shape)
  return image_resized/256., image_decoded.shape 

#%%
dataset = tf.data.Dataset.from_tensor_slices((ll))
dataset,shapes = dataset.map(_parse_function)
batch_dataset = dataset.batch(10)

    
iterator = batch_dataset.make_one_shot_iterator()

images,originals = iterator.get_next()




#%%

