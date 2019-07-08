# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:12:51 2017

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
import numpy as np
import helper
import sys

from read_ontology import read_ontology


#%%
def val_nn(sess, batch_size, 
             dataset_file, image_shape, num_classes,
             cross_entropy_loss, input_image,
             corr_label, keep_prob, checkpoint_num):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    
    save_net = '/media/avarfolomeev/storage/Data/Segmentation/UK/nets/OS_44c';
    min_loss_file = '/media/avarfolomeev/storage/Data/Segmentation/UK/nets/min_loss.txt';
    lr_file = '/media/avarfolomeev/storage/Data/Segmentation/UK/nets/lr.txt';
    
        
    get_val_batches_fn = build_batch_fn(dataset_file, 'val',
                                            image_shape, num_classes);
    bnum = 0
    #read updated lr from the file
    bnum = 0
    
    cum_loss = 0
    intersect = np.zeros(num_classes)
    union = np.zeros(num_classes)

    sess.run(tf.local_variables_initializer())

    for image, label in get_val_batches_fn(batch_size):
        image = image[:batch_size]
        label = label[:batch_size]


        #sess.run([conf_matrix], feed_dict={input_image:image, 
        #                                    corr_label:label,
        #                                    keep_prob:1})
        val_loss, _intersect, _union = sess.run([cross_entropy_loss, t_intersect, t_union],
                                    feed_dict={input_image:image, 
                                            corr_label:label,
                                            keep_prob:1})
        intersect = intersect + _intersect
        union = union + _union

        sys.stdout.write('\rVal ' + str(bnum) + '  ' + str(val_loss) + '   \r')
        sys.stdout.flush()      
        cum_loss += val_loss
        bnum = bnum + 1    
            
    val_loss = cum_loss / bnum
    
    
    union[union < 1] = 1
    IoU = intersect / union
    
    
    fn =  save_net + '-' + str(checkpoint_num) + '.' + str(val_loss)
    with open(fn,"w") as f:
        for cls in range(num_classes):
            f.write("%15.15s, %.2f\n" % (ont[cls].name, IoU[cls]) )
                    
           
            
#%%
def build_batch_fn(_dataset_descriptor, _split, _image_shape, _num_classes):
    try:
        _base_dir = open(_dataset_descriptor,'rt').readline().rstrip('\n')
    except:    
        _base_dir = '/media/avarfolomeev/storage/Data/Segmentation/UK/UK-4'
    
    
    _fn = helper.gen_batch_function(_base_dir,
                                   _split,_image_shape, _num_classes)
    return (_fn)

            
            
#%%

#def retrain():
    
tf.reset_default_graph()

dataset_file = '/media/avarfolomeev/storage/Data/Segmentation/UK/dataset.txt'
timestamp = time.strftime("%Y%m%d_%H%M%S");


ont, colors = read_ontology('/media/avarfolomeev/storage/Data/Segmentation/UK/UK-4/Ontology F8.csv')
num_classes = len(ont)
image_shape=(768,960)

batch_size = 4

checkpoint_num = 188

alfa = (127,) #semi-transparent

config = tf.ConfigProto(
   gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95),
   device_count = {'GPU': 1}
)
sess = tf.Session(config = config)

sess.run(tf.global_variables_initializer())

load_net = '/media/avarfolomeev/storage/Data/Segmentation/UK/nets/OS_44c-' + str(checkpoint_num)

min_loss_name = 'min_loss.txt'

saver = tf.train.import_meta_graph(load_net + '.meta')
saver.restore(sess,load_net)


model = tf.get_default_graph()

input_image = model.get_tensor_by_name('image_input:0')
keep_prob = model.get_tensor_by_name('keep_prob:0')
nn_output = model.get_tensor_by_name('layer3_up/BiasAdd:0')
correct_label = model.get_tensor_by_name('correct_label:0')


assert(nn_output.shape[-1] == num_classes)

labels = tf.reshape(correct_label, [-1,num_classes])
labels0 = labels[:,0];

class_filter = tf.squeeze(tf.where(tf.not_equal(labels0,1)),1)


logits = tf.reshape(nn_output,(-1,num_classes))


gt = tf.gather(labels,class_filter)
prediction = tf.gather(logits,class_filter)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=gt,
                                                        logits = prediction,
                                                        name = "cross-ent")
cross_entropy_loss = tf.reduce_mean(cross_entropy);

y_true = tf.math.argmax(gt,axis=-1)
y_pred = tf.math.argmax(prediction , axis=-1)

miou, conf_matrix = tf.metrics.mean_iou(y_true, y_pred, num_classes)
sum_true = tf.reduce_sum(conf_matrix,axis=0)
sum_pred = tf.reduce_sum(conf_matrix, axis = 1)
t_intersect = tf.diag_part(conf_matrix)
t_union = sum_true + sum_pred - t_intersect


print('validating')
val_nn(sess, batch_size, 
         dataset_file, image_shape, num_classes, cross_entropy_loss,
         input_image, correct_label, keep_prob, checkpoint_num) 




#if __name__ == '__main__':
#    retrain()
           
