# -*- coding: utf-8 -*-
"""
inference with SegmentationSuite networks using tf.datasets api

"""

import numpy as np
import os.path
import scipy.misc
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import labels_vox as lbl

import matplotlib.pyplot as plt
import sys
from read_ontology import read_ontology

import argparse





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

base_dir = '/media/avarfolomeev/storage/Data/'

try:
    base_dir = os.environ['BASE_DATA_PATH']
except:
    pass
        

input_folder = base_dir + '/Voxels/201904_UK/'
put_folder = base_dir + '/Voxels/201904_UK/o/'
dataname = 'data/'
ride = '20190403_064438'  #'20190216_115220'
camera = 'argus_cam_0'
ontology, _ = read_ontology(base_dir+'/Segmentation/UK/1375272-ontology.csv')


image_shape= (768,960) # (1024,1216)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default= base_dir + '/Segmentation/UK/nets/BiSeNet_ResNet101/0290/', required=False, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--dataset', type=str, default= base_dir + "/Voxels/201904_UK", required=False, help='The dataset you are using')
parser.add_argument('--output', type=str, default=base_dir + "/Voxels/201904_UK", required=False, help='The dataset you are using')
parser.add_argument('--ride', type=str, default="20190403_064438", required=False, help='ride name')
parser.add_argument('--camera',type=str, default="argus_cam_0", required=False, help='camera name')
parser.add_argument('--start',type=int, default=0, required=False, help='start from')
parser.add_argument('--end',type=int, default=None, required=False, help='process to')

args = parser.parse_args()


input_folder = args.dataset;
out_folder = args.output;
load_net = args.checkpoint_path+'model.ckpt'

data_folder = os.path.join(input_folder,ride, camera)
camera_folder = os.path.join(out_folder,ride,camera)

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


num_classes = len(ontology)

alfa = (127,) #semi-transparent
colors = np.array([ont.color + alfa for ont in ontology]).astype(np.uint8)




print("reading from  ", os.path.join(data_folder, dataname));

l = glob(os.path.join(data_folder, dataname, '*.jpg'))
l.sort()
print(len(l), " files read");

if (len(l) == 0):
	exit();

total_num = len(l)

l = l[args.start:args.end]


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


print ("Using net ", load_net)

meta = load_net + '.meta'


csvname = os.path.join(out_folder, ride, ride + "_" + camera + ".csv")

one = tf.constant(1.0, dtype = float)
restorer = tf.train.import_meta_graph(meta, input_map = {'IteratorGetNext:0':images})

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

restorer.restore(sess,load_net)

nn_output = tf.get_default_graph().get_tensor_by_name('logits/BiasAdd:0') #layer3_up/BiasAdd:0'

#logits = tf.reshape(nn_output,(-1,num_classes))

softmax = tf.nn.softmax(nn_output,3)
argmax = tf.math.argmax(softmax, axis=3)
argmax = tf.expand_dims(argmax,-1)

#resized = tf.image.resize_images(argmax, image0.get_shape(), #(1028,1232), 
#                                 tf.image.ResizeMethod.NEAREST_NEIGHBOR);

cnt = args.start
use_subdirs = False

with  open(csvname,"a") as csvfile:

    while (True):
        try:
            out,names,im0 = sess.run([argmax,filenames,original_images])
            out_colors = colors[out[0,:,:,0]]    
            original_shape = im0[0].shape[1::-1]
            out_colors = cv2.resize(out_colors, original_shape, interpolation=cv2.INTER_NEAREST)
            colors_img = scipy.misc.toimage(out_colors, mode="RGBA")
            overlay_im = scipy.misc.toimage(im0[0])

            overlay_im.paste(colors_img,box=None,mask=colors_img)
            
            out_file = names[0].decode('utf-8').replace(dataname,overlay_name).replace(data_folder, camera_folder)
            scipy.misc.imsave(out_file, overlay_im)
    
            for idx in range (len(names)):
                
                im_name = names[idx].decode('utf-8')
                out_file = im_name.replace(dataname,road_name).replace('.jpg','.png').replace(data_folder, camera_folder)
                out_name = os.path.split(out_file)[1]
                _str = os.path.split(im_name)[1] + ", " + out_name + '\n'
                csvfile.write(_str)
                
                mx = out[idx,:,:,0]
    

                original_shape = im0[idx].shape[1::-1]
                mx = cv2.resize(mx,original_shape, interpolation=cv2.INTER_NEAREST)

                cv2.imwrite(out_file,mx)
                print(cnt, " from", total_num, " ", out_name)
                cnt = cnt + 1            
            
        except tf.errors.OutOfRangeError:
            break;
    csvfile.close();        
#%%
