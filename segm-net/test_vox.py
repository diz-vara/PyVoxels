# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 09:29:12 2017

@author: avarfolomeev
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

#%%
#saver = tf.train.import_meta_graph('./exports/KITTI_segm/KITTI_segm-33.meta')
#saver.restore(sess,'./exports/KITTI_segm/KITTI_segm-33')

labels = lbl.labels_vox
num_classes = len(labels)

alfa = (127,) #semi-transparent
colors = np.array([label.color + alfa for label in labels]).astype(np.uint8)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

base_dir = '/media/undead/Data'

load_net = base_dir + '/Segmentation/vox/vox-net-5837'

saver = tf.train.import_meta_graph(load_net + '.meta')
saver.restore(sess,load_net)


model = tf.get_default_graph()

input_image = model.get_tensor_by_name('image_input:0')
keep_prob = model.get_tensor_by_name('keep_prob:0')
nn_output = model.get_tensor_by_name('layer3_up/BiasAdd:0')


logits = tf.reshape(nn_output,(-1,num_classes))


#%%

def segment_files(image_files):
    
    images = []
    shapes = []
    for image_file in image_files:
        image0 = scipy.misc.imread(image_file)
        original_shape = image0.shape    
    
        image = scipy.misc.imresize(image0, image_shape)
        images.append(image)
        shapes.append(original_shape)

    #image = cv2.GaussianBlur(image,(3,3),2)
    batch_size = len(images)

    b_out_shape = (batch_size,image_shape[0], image_shape[1], num_classes)
    
    im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, input_image: images})

    out_ims=[]
    b_res = im_softmax[0].reshape(b_out_shape)
    for res in range(batch_size):
        mx=np.argmax(b_res[res],2)
        original_shape = shapes[res]
        out_colors = colors[mx]    

        out_image = cv2.resize(mx, (original_shape[1], original_shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
        out_ims.append(out_image)

        if (res == 0):
            #out_colors = scipy.misc.imresize(out_colors,original_shape)
            colors_img = scipy.misc.toimage(out_colors, mode="RGBA")
        
            street_im = scipy.misc.toimage(images[res])
            street_im.paste(colors_img,box=None,mask=colors_img)
            street_im = scipy.misc.imresize(street_im,original_shape)
            
        
    return street_im, out_ims

#%%
def segment_file(image_file):
    image0 = scipy.misc.imread(image_file)
    original_shape = image0.shape

    image = scipy.misc.imresize(image0, image_shape)

    #image = cv2.GaussianBlur(image,(3,3),2)
    
    im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, input_image: [image]})

    res = im_softmax[0].reshape(out_shape)
    mx=np.argmax(res,2)

    out_colors = colors[mx]    

    #out_image = scipy.misc.toimage(mx, mode = 'L')
    out_image = cv2.resize(mx, (original_shape[1], original_shape[0]), 
                                interpolation=cv2.INTER_NEAREST)

    out_colors = scipy.misc.imresize(out_colors,original_shape)
    colors_img = scipy.misc.toimage(out_colors, mode="RGBA")

    street_im = scipy.misc.toimage(image0)
    street_im.paste(colors_img,box=None,mask=colors_img)
    return street_im, out_image

#plt.imshow(street_im)
#%%
dataset = '8Tb'

if dataset == 'London':
    data_folder='/media/avarfolomeev/storage/Data/voxels/2018_03_08/L21/'
    image_shape=(384,640)
    dataname = 'image_02/data/'
    l = glob(os.path.join(data_folder, dataname, '*.png'))
elif dataset == 'CN':
    data_folder='/media/avarfolomeev/storage/Data/CN/2017_11_08_06_32_41/'
    image_shape=(352,640)
    dataname = 'Collected/'
    l = glob(os.path.join(data_folder, dataname, '*.jpg'))
elif dataset == 'KITTI':
    #data_folder='/media/D/DIZ/Datasets/KITTI/2011_10_03/2011_10_03_drive_0027_sync/'
    data_folder='/media/D/DIZ/Datasets/KITTI/2011_09_26/2011_09_26_drive_0084_sync/'
    image_shape=(192,608)
    dataname = 'image_02/data/'
    l = glob(os.path.join(data_folder, dataname, '*.png'))
elif dataset == 'CS':
    #data_folder='/media/D/DIZ/Datasets/KITTI/2011_10_03/2011_10_03_drive_0027_sync/'
    data_folder='/media/D/DIZ/Datasets/KITTI/2011_09_26/2011_09_26_drive_0084_sync/'
    image_shape=(320,640)
    dataname = 'image_02/data/'
    l = glob(os.path.join(data_folder, dataname, '*.png'))
elif dataset == 'spb-R1':
    data_folder='/media/avarfolomeev/storage/Data/voxels/20180525/ride01/'
    image_shape=(384,640)
    dataname = 'image_02/data/'
    l = glob(os.path.join(data_folder, dataname, '*.png'))
elif dataset == 'NavInfo':
    data_folder='/media/avarfolomeev/storage/Data/voxels/NavInfo/'
    image_shape=(384,640)
    dataname = 'Screens/'
    l = glob(os.path.join(data_folder, dataname, '*.png'))
elif dataset == 'Data':
    data_folder='/media/undead/Data/Voxels/out/201809_usa/test7_2_b/argus_cam_5/'
    image_shape=(576,1024)
    dataname = 'data/'
    l = glob(os.path.join(data_folder, dataname, '*.jpg'))
elif dataset == '8Tb':
    data_folder='/media/undead/8Tb/out'
    image_shape=(576,1024)
    dataname = 'data/'
#    l = glob(os.path.join(data_folder, dataname, '*.jpg'))
elif dataset == 'Work':
    data_folder='/media/undead/Work/Voxels/out/'
    image_shape=(576,1024)
    dataname = 'data/'
    l = glob(os.path.join(data_folder, dataname, '*.jpg'))
    


base_dir = '/media/undead/8Tb/'
data_folder= base_dir + '/out/'


ride = 'test11_2'
camera = 'argus_cam_5'

data_folder = os.path.join(data_folder,ride, camera)
out_folder = os.path.join('/media/undead/ssd/Voxels/',ride,camera)
l = glob(os.path.join(data_folder, dataname, '*.jpg'))


road_name = 'Xroad_v'
overlay_name = 'Xoverlay_v'

try:
    os.makedirs(os.path.join(out_folder,road_name))
except:
    pass        

try:
    os.makedirs(os.path.join(out_folder,overlay_name))
except:
    pass        


out_shape = (image_shape[0], image_shape[1], num_classes)

#%%
batch_size = 10
cnt = 0
num = len(l)
start  = 0
end = 12540
flist = []



if (end < 0 or end > num):
    end = num;

csvname = os.path.join(out_folder, ride + "_" + camera + "_0" + ".csv")
 

with  open(csvname,"w") as csvfile:
    for cnt in range(start,end):
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
            
        im_file = l[cnt]
        flist.append(im_file)

        out_file = im_file.replace(dataname,road_name+sub_dir).replace('.jpg','.png').replace(data_folder, out_folder)
        out_name = sub_dir + os.path.split(out_file)[1]
        _str = os.path.split(im_file)[1] + ", " + out_name + '\n'
        csvfile.write(_str)


    csvfile.close();            
#%%

"""
        print(cnt, " from", num, " ", im_file)
        
        if (len(flist) >= batch_size or cnt == end-1):
            im_out, masks = segment_files(flist)
            out_file = flist[0].replace(dataname,overlay_name + sub_dir).replace(data_folder,out_folder)


            scipy.misc.imsave(out_file, im_out)
            for idx in range (len(flist)):
                out_file = flist[idx].replace(dataname,road_name + sub_dir).replace('.jpg','.png').replace(data_folder, out_folder)
                cv2.imwrite(out_file,masks[idx])
            flist = []    
"""            
