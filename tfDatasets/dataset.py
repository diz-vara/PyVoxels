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
import labels_vox as lbl
import numpy as np
import helper

import matplotlib.pyplot as plt
import sys
from read_ontology import read_ontology



#%%
def _parse_function(filename):
  try:  
      image_string = tf.read_file(filename)
      image_decoded = tf.image.decode_jpeg(image_string)
      image_resized = tf.image.resize_images(image_decoded, image_shape)
  except:
      print("Error reading file ", filename)
      image_resized = None
      image_decoded = None
  return image_resized, filename, image_decoded

#%%

base_dir = '/media/nvidia/'

data_folder= base_dir + '/8Tb/201902_USA/out'
out_folder = base_dir + '/ssd/Voxels'

image_shape= (1024,1216)
dataname = 'data/'

ride = '20190216_115220'
camera = 'argus_cam_0'

ontology, _ = read_ontology('/media/nvidia/Data/Segmentation/UK/1375272-ontology.csv')


nArg = len(sys.argv)
if (nArg > 4):
  data_folder = sys.argv[1]
  out_folder = sys.argv[2]
  ride = sys.argv[3]
  camera = sys.argv[4]
elif (nArg > 2):  
  ride = sys.argv[1]
  camera = sys.argv[2]

data_folder = os.path.join(data_folder,ride, camera)
camera_folder = os.path.join(out_folder,ride,camera)

print("reading from  ", os.path.join(data_folder, dataname));

l = glob(os.path.join(data_folder, dataname, '*.jpg'))
l.sort()
print(len(l), " files read");

if (len(l) == 0):
	exit();

start = 0000
if (nArg > 5):
    try:
        start = int(sys.argv[5])
    except:
        start = 0;    

load_net = ""

if (nArg > 6):
    load_net = sys.argv[6]


total_num = len(l)

l = l[start:]
#l = l[5::50] 

road_name = 'Xroad/'
overlay_name = 'Xoverlay/'

try:
    os.makedirs(os.path.join(camera_folder,road_name))
except:
    pass        

try:
    os.makedirs(os.path.join(camera_folder,overlay_name))
except:
    pass        


print("writing to ", os.path.join(camera_folder, road_name));


labels = lbl.labels_vox
num_classes = len(ontology)

alfa = (127,) #semi-transparent
colors = np.array([ont.color + alfa for ont in ontology]).astype(np.uint8)
#%%
batchsize = 5



tf.reset_default_graph();

dataset = tf.data.Dataset.from_tensor_slices((l))
dataset = dataset.map(_parse_function)
batch_dataset = dataset.batch(batchsize)

    
iterator = batch_dataset.make_one_shot_iterator()

images,filenames,original_images = iterator.get_next()

image0=original_images[0]

#kernel = np.ones((batchsize,1,1,3))
#kernel[0,0,0,0] = 0.9
#images_adj = tf.nn.conv2d(images, kernel,[1,1,1,1],'SAME')


#load_net = base_dir + 'Data/Segmentation/net/my2-net-73949'
#load_net = base_dir + 'Data/Segmentation/vox/vox-net-lp-6058'

if (len(load_net) == 0):
	load_net = base_dir + "Data/Segmentation/UK/nets/OS_net-86"


print ("Using net ", load_net)

meta = load_net + '.meta'


csvname = os.path.join(out_folder, ride, ride + "_" + camera + ".csv")

one = tf.constant(1.0, dtype = float)
restorer = tf.train.import_meta_graph(meta, input_map = {'image_input:0':images, 'keep_prob:0':one })

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

restorer.restore(sess,load_net)

nn_output = tf.get_default_graph().get_tensor_by_name('layer3_up/BiasAdd:0')

#logits = tf.reshape(nn_output,(-1,num_classes))

softmax = tf.nn.softmax(nn_output,3)
argmax = tf.math.argmax(softmax, axis=3)
argmax = tf.expand_dims(argmax,-1)

#resized = tf.image.resize_images(argmax, image0.get_shape(), #(1028,1232), 
#                                 tf.image.ResizeMethod.NEAREST_NEIGHBOR);

cnt = start
use_subdirs = False

with  open(csvname,"a") as csvfile:

    while (True):
        try:
            if (use_subdirs):
                if (cnt%2000 == 0 or  not 'sub_dir' in locals() or sub_dir is None):
                    sub_dir = "/{:05d}/".format(cnt-cnt%2000)
                    try:
                        os.makedirs(os.path.join(camera_folder,overlay_name + sub_dir))
                    except:
                        pass    
                    try:
                        os.makedirs(os.path.join(camera_folder,road_name + sub_dir))
                    except:
                        pass
            else:
                sub_dir = '';
        
    
            out,names,im0 = sess.run([argmax,filenames,original_images])
            out_colors = colors[out[0,:,:,0]]    
            original_shape = im0[0].shape[1::-1]
            out_colors = cv2.resize(out_colors, original_shape, interpolation=cv2.INTER_NEAREST)
            colors_img = scipy.misc.toimage(out_colors, mode="RGBA")
            overlay_im = scipy.misc.toimage(im0[0])

            #print(im0[0].shape, out_colors.shape)
            
            overlay_im.paste(colors_img,box=None,mask=colors_img)
            
            
            out_file = names[0].decode('utf-8').replace(dataname,overlay_name+sub_dir).replace(data_folder, camera_folder)
            scipy.misc.imsave(out_file, overlay_im)
    
            for idx in range (len(names)):
                
                im_name = names[idx].decode('utf-8')
                out_file = im_name.replace(dataname,road_name+sub_dir).replace('.jpg','.png').replace(data_folder, camera_folder)
                out_name = sub_dir + os.path.split(out_file)[1]
                _str = os.path.split(im_name)[1] + ", " + out_name + '\n'
                csvfile.write(_str)
                
                mx = out[idx,:,:,0]
    
                if ( 'indexes_0' in globals() and len(indexes_0) >= len(colors)):
                    mx = indexes_0a[mx]

                original_shape = im0[idx].shape[1::-1]
                mx = cv2.resize(mx,original_shape, interpolation=cv2.INTER_NEAREST)

                cv2.imwrite(out_file,mx)
                print(cnt, " from", total_num, " ", out_name)
                cnt = cnt + 1            
            
        except tf.errors.OutOfRangeError:
            break;
    csvfile.close();        
#%%

