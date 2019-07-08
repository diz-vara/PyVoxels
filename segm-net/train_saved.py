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
def train_nn(sess, epochs, batch_size, 
             dataset_file, image_shape, num_classes,
             train_op, cross_entropy_loss, input_image,
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
    
    save_net = '/media/avarfolomeev/storage/Data/Segmentation/UK/nets/OS_44c';
    min_loss_file = '/media/avarfolomeev/storage/Data/Segmentation/UK/nets/min_loss.txt';
    lr_file = '/media/avarfolomeev/storage/Data/Segmentation/UK/nets/lr.txt';
    
    #sess.run(tf.global_variables_initializer())
    #saver = tf.train.Saver();

    #lr = sess.run(learning_rate)
    merged = tf.summary.merge_all()
    lr = 1.e-4
    min_loss = 0.4
    for epoch in range (epochs):
        print ('epoch {} ({}) '.format(epoch+base, epoch))
        sys.stdout.flush()
        
        get_train_batches_fn = build_batch_fn(dataset_file, 'train',
                                              image_shape, num_classes);
        
        get_val_batches_fn = build_batch_fn(dataset_file, 'val',
                                              image_shape, num_classes);
        bnum = 0
        #read updated lr from the file
        try:
            lr = float(open(lr_file).read())
        except:
            0;    
        _lr = lr #* 0.9965                     
        print(" LR = {:g}".format(lr))
        open(lr_file,'w').write(str(_lr))
        cum_loss = 0
        for image, label in get_train_batches_fn(batch_size):
            image = image[:batch_size]
            label = label[:batch_size]
            t_summary, _, train_loss = sess.run([train_summary, train_op, cross_entropy_loss],
                                     feed_dict={input_image:image, 
                                                corr_label:label,
                                                keep_prob:0.5, learning_rate:lr})
            cum_loss += train_loss
            sys.stdout.write('\rTrain ' + str(bnum) + '  ' + str(train_loss) + '   \r')
            sys.stdout.flush()      
            bnum = bnum + 1  
        train_loss = cum_loss / bnum
        sys.stdout.write("\r\nTrain loss: " + str(train_loss) + '\r\n')
        sys.stdout.flush()      
        bnum = 0
        
        cum_loss = 0
        intersect = np.zeros(num_classes)
        union = np.zeros(num_classes)
        for image, label in get_val_batches_fn(batch_size):
            image = image[:batch_size]
            label = label[:batch_size]
            v_summary, val_loss, _intersect, _union = sess.run([val_summary, cross_entropy_loss, t_intersect, t_union],
                                     feed_dict={input_image:image, 
                                                corr_label:label,
                                                keep_prob:1, learning_rate:lr})
            intersect = intersect + _intersect
            union = union + _union

            sys.stdout.write('\rVal ' + str(bnum) + '  ' + str(val_loss) + '   \r')
            sys.stdout.flush()      
            cum_loss += val_loss
            bnum = bnum + 1    
               
        val_loss = cum_loss / bnum
        
        writer.add_summary(t_summary, epoch)
        writer.add_summary(v_summary, epoch)
        #writer.add_summary(merged, epoch)
        print("\r\nLoss = {:g} {:g}".format(train_loss, val_loss))     
        print()
        
        union[union < 1] = 1
        IoU = intersect / union
        
        #for cls in range(num_classes):
        #    print(ont[cls].name, IoU[cls])
        #print()



        # read updated min_loss from the file
        try:
            min_loss = float(open(min_loss_file).read())
        except:
            pass;    
        
        if (val_loss < min_loss):
            print("saving at step {:d}".format(epoch+base))     
            min_loss = val_loss;
            saver.save(sess, save_net,
                       global_step=epoch+base)
            #save empty file with loss value           
            fn =  save_net + '-' + str(epoch+base) + '.' + str(val_loss)
            with open(fn,"w") as f:
                for cls in range(num_classes):
                    f.write("%15.15s, %.2f\n" % (ont[cls].name, IoU[cls]) )
                    

                                        
            open(min_loss_file,'w').write(str(min_loss))
           
            
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

epochs = 5000
batch_size = 4


alfa = (127,) #semi-transparent

config = tf.ConfigProto(
   gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95),
   device_count = {'GPU': 1}
)
sess = tf.Session(config = config)

#saver = tf.train.Saver()

load_net = '/media/avarfolomeev/storage/Data/Segmentation/UK/nets/OS_44c-1006'

min_loss_name = 'min_loss.txt'

saver = tf.train.import_meta_graph(load_net + '.meta')
saver.restore(sess,load_net)


model = tf.get_default_graph()

input_image = model.get_tensor_by_name('image_input:0')
keep_prob = model.get_tensor_by_name('keep_prob:0')
nn_output = model.get_tensor_by_name('layer3_up/BiasAdd:0')
correct_label = model.get_tensor_by_name('correct_label:0')
learning_rate = model.get_tensor_by_name('learning_rate:0')


assert(nn_output.shape[-1] == num_classes)

labels = tf.reshape(correct_label, [-1,num_classes])
labels0 = labels[:,0];

class_filter = tf.squeeze(tf.where(tf.not_equal(labels0,1)),1)


logits = tf.reshape(nn_output,(-1,num_classes))

#print(correct_label.get_shape(), nn_output.get_shape())


gt = tf.gather(labels,class_filter)
prediction = tf.gather(logits,class_filter)

#print(gt.get_shape(), prediction.get_shape())


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=gt,
                                                        logits = prediction,
                                                        name = "cross-ent")
cross_entropy_loss = tf.reduce_mean(cross_entropy);

y_true = tf.math.argmax(gt,axis=-1)
y_pred = tf.math.argmax(prediction , axis=-1)

conf_matrix = tf.confusion_matrix(y_true, y_pred, num_classes)
sum_true = tf.reduce_sum(conf_matrix,axis=0)
sum_pred = tf.reduce_sum(conf_matrix, axis = 1)
t_intersect = tf.diag_part(conf_matrix)
t_union = sum_true + sum_pred - t_intersect


#tf.summary.scalar('cross_entropy', cross_entropy)
train_summary = tf.summary.scalar('train_loss', cross_entropy_loss)
val_summary = tf.summary.scalar('val_loss', cross_entropy_loss)
#tf.summary.scalar('val loss', val_loss)
lr_summary = tf.summary.scalar('learning_rate', learning_rate)


train_op=model.get_collection('train_op')[0]

train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'layer3')
#train_op = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss, var_list = train_vars)
#train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss, var_list = train_vars)


writer = tf.summary.FileWriter('/media/avarfolomeev/storage/Data/Segmentation/UK/logs')

print('training')
train_nn(sess, epochs, batch_size, 
         dataset_file, image_shape, num_classes,
         train_op,
         cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate, 1007) 




#if __name__ == '__main__':
#    retrain()
           
