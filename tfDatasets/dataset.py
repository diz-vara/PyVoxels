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




#%%
def _parse_function(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, image_shape)
  return image_resized/256., filename, image_decoded

#%%

base_dir = '/media/undead/'

data_folder= base_dir + '/8Tb/out'
out_folder = base_dir + '/ssd/Voxels'

image_shape=(576,1024)
dataname = 'data/'

ride = 'test11_2'
camera = 'argus_cam_5'

data_folder = os.path.join(data_folder,ride, camera)
out_folder = os.path.join(out_folder,ride,camera)
l = glob(os.path.join(data_folder, dataname, '*.jpg'))



start = 30000
total_num = len(l)

l = l[start:]
#l = l[5::50] 

road_name = 'Xroad_v/'
overlay_name = 'Xoverlay_v/'

try:
    os.makedirs(os.path.join(out_folder,road_name))
except:
    pass        

try:
    os.makedirs(os.path.join(out_folder,overlay_name))
except:
    pass        


labels = lbl.labels_vox_0
num_classes = len(labels)

alfa = (127,) #semi-transparent
colors = np.array([label.color + alfa for label in labels]).astype(np.uint8)
#%%
batchsize = 12

tf.reset_default_graph();

dataset = tf.data.Dataset.from_tensor_slices((l))
dataset = dataset.map(_parse_function)
batch_dataset = dataset.batch(batchsize)

    
iterator = batch_dataset.make_one_shot_iterator()

images,filenames,original_images = iterator.get_next()

image0=original_images[0]


#load_net = base_dir + 'Data/Segmentation/net/my2-net-73949'
load_net = base_dir + 'Data/Segmentation/vox/vox-net-5837'

meta = load_net + '.meta'


csvname = os.path.join(out_folder, ride + "_" + camera + ".csv")

one = tf.constant(1.0, dtype = float)
restorer = tf.train.import_meta_graph(meta, input_map = {'image_input:0':images*256, 'keep_prob:0':one })

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

restorer.restore(sess,load_net)
nn_output = tf.get_default_graph().get_tensor_by_name('layer3_up/BiasAdd:0')

#logits = tf.reshape(nn_output,(-1,num_classes))

softmax = tf.nn.softmax(nn_output,3)
argmax = tf.math.argmax(softmax, axis=3)
argmax = tf.expand_dims(argmax,-1)

resized = tf.image.resize_images(argmax,(720,1280), 
                                 tf.image.ResizeMethod.NEAREST_NEIGHBOR);

cnt = start

with  open(csvname,"a") as csvfile:

    while (True):
        try:
            if (cnt%2000 == 0 or  not 'sub_dir' in locals() or sub_dir is None):
                sub_dir = "/{:05d}/".format(cnt-cnt%2000)
                try:
                    os.makedirs(os.path.join(out_folder,overlay_name + sub_dir))
                except:
                    pass    
                try:
                    os.makedirs(os.path.join(out_folder,road_name + sub_dir))
                except:
                    pass
        
    
            out,names,im0 = sess.run([resized,filenames,image0])
            out_colors = colors[out[0,:,:,0]]    
    
            colors_img = scipy.misc.toimage(out_colors, mode="RGBA")
            overlay_im = scipy.misc.toimage(im0)
            overlay_im.paste(colors_img,box=None,mask=colors_img)
            
            
            out_file = names[0].decode('utf-8').replace(dataname,overlay_name+sub_dir).replace(data_folder, out_folder)
            scipy.misc.imsave(out_file, overlay_im)
    
            for idx in range (len(names)):
                
                im_name = names[idx].decode('utf-8')
                out_file = im_name.replace(dataname,road_name+sub_dir).replace('.jpg','.png').replace(data_folder, out_folder)
                out_name = sub_dir + os.path.split(out_file)[1]
                _str = os.path.split(im_name)[1] + ", " + out_name + '\n'
                csvfile.write(_str)
                
                mx = out[idx,:,:,0]
    
                if ( 'indexes_0' in globals() and len(indexes_0) >= len(colors)):
                    mx = indexes_0a[mx]


                cv2.imwrite(out_file,mx)
                print(cnt, " from", total_num, " ", out_name)
                cnt = cnt + 1            
            
        except tf.errors.OutOfRangeError:
            break;
    csvfile.close();        
#%%

