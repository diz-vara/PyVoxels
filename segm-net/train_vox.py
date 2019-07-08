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
import labels_diz as lbl_diz
import labels_vox as lbl_vox

import numpy as np
import helper
import sys


#%%
def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    
    result = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                            logits = result,
                                                            name = "cross-ent")
    loss = tf.reduce_mean(cross_entropy);
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate) 

    train_op = optimizer.minimize(loss)
    
    return result, train_op, loss
#%%
def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             corr_label, keep_prob, learning_rate, base = 0):
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
    
    save_net = '/media/avarfolomeev/storage/Data/Segmentation/vox_segm/vox-net-LP2';
    #sess.run(tf.global_variables_initializer())
    #saver = tf.train.Saver();

    #lr = sess.run(learning_rate)
    merged = tf.summary.merge_all()
    lr = 3e-4
    min_loss = 0.2
    for epoch in range (epochs):
        print ('epoch {}  '.format(epoch))
        print(" LR = {:g}".format(lr))
        sys.stdout.flush()
        bnum = 0
        for image, label in get_batches_fn(batch_size):
            image = image[:batch_size]
            label = label[:batch_size]
            summary, loss = sess.run([train_op, cross_entropy_loss],
                                     feed_dict={input_image:image, 
                                                corr_label:label,
                                     keep_prob:0.5, learning_rate:lr})
            sys.stdout.write('\r' + str(bnum) + '  ' + str(loss) + '   \r')
            sys.stdout.flush()      
            bnum = bnum + 1                   
        #writer.add_summary(summary, epoch)                          
        lr = lr * 0.997                     
        print(" Loss = {:g}".format(loss))     
        print()                        
        if (loss < min_loss):
            print("saving at step {:d}".format(epoch+base))     
            min_loss = loss;
            saver.save(sess, save_net,
                       global_step=epoch+base)
            #save empty file with loss value           
            fn =  save_net + '-' + str(epoch+base) + '.' + str(loss)
            f = open(fn,"wb")
            f.close()
#%%

#def retrain():
    
tf.reset_default_graph()

new_graph = tf.Graph()

data_dir = '/media/avarfolomeev/storage/Data/Segmentation/data'
runs_dir = '/media/avarfolomeev/storage/Data/Segmentation/vox_segm/runs'
timestamp = time.strftime("%Y%m%d_%H%M%S");

export_dir = '/media/avarfolomeev/storage/Data/Segmentation/vox_segm/exports/' + timestamp;

labels = lbl_vox.labels_vox
#ids = [l.id for l in labels]


num_classes = 44 #len(labels)

image_shape=(768,960)

epochs = 5000
batch_size = 4


alfa = (127,) #semi-transparent
colors = np.array([label.color + alfa for label in labels]).astype(np.uint8)

config = tf.ConfigProto(
   gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8),
   device_count = {'GPU': 1}
)
sess = tf.Session(config = config)

#saver = tf.train.Saver()

load_net = '/media/avarfolomeev/storage/Data/Segmentation/vox/vox-net-3-9307'


saver = tf.train.import_meta_graph(load_net + '.meta', tf.trainable_variables())
writer = tf.summary.FileWriter('/media/avarfolomeev/storage/Data/Segmentation/UK/logs')

model = tf.get_default_graph()


layer3_conv1 = model.get_tensor_by_name('layer_3_conv1/Relu:0')

layer3_up = tf.layers.conv2d_transpose(layer3_conv1, num_classes, 10,
                                         strides = (8,8),
                                         padding = 'same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                         name = 'layer3_up')

init_layer3_op = tf.initialize_variables(tf.trainable_variables())

saver.restore(sess,load_net)

input_image = model.get_tensor_by_name('image_input:0')
keep_prob = model.get_tensor_by_name('keep_prob:0')
nn_output = layer3_up; #model.get_tensor_by_name('layer3_up/BiasAdd:0')
#correct_label = model.get_tensor_by_name('correct_label:0')
correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes],
                                       name = 'correct_label')


learning_rate = model.get_tensor_by_name('learning_rate:0')


assert(nn_output.shape[-1] == num_classes)


logits = tf.reshape(nn_output,(-1,num_classes))

labels = tf.reshape(correct_label, (-1, num_classes))

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                        logits = logits,
                                                        name = "cross-ent")
loss = tf.reduce_mean(cross_entropy);
#optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate) 
#train_op = optimizer.minimize(loss)



#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits = logits, name = "cross-ent")
#loss = tf.reduce_mean(cross_entropy);

#logits, train_op, loss = optimize(nn_output, correct_label, 
#                                  learning_rate, num_classes)


get_batches_fn = helper.gen_batch_function('/media/avarfolomeev/storage/Data/Segmentation/UK/UK-4', 'train',
                                           image_shape, num_classes)






train_op=model.get_collection('train_op')[0]

train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'layer3')
#train_op = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss, var_list = train_vars)
#train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss, var_list = train_vars)

sess.run(init_layer3_op)



print('training')
train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
         loss, input_image, correct_label, keep_prob, learning_rate, 0) 




#if __name__ == '__main__':
#    retrain()
            