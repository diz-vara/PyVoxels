# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:12:51 2017

@author: avarfolomeev
"""

import numpy as np
import os.path
import time
import tensorflow as tf
import numpy as np
import sys

from read_ontology import read_ontology
from train import *

import argparse


base_dir = '/media/avarfolomeev/storage/Data/'

try:
    base_dir = os.environ['BASE_DATA_PATH']
except:
    pass

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
parser.add_argument('--continue_training', type=str2bool, default='no', help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="UK-4", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=768, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=960, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=2, help='Number of images in each batch')
parser.add_argument('--model_name', type=str, default="OS-44-new", help='Model name')
parser.add_argument('--checkpoint_num', type=int, default=0, help='checkpoint num to load')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--prefix', type=str, default="", help='optional prefix to separate n/w')
parser.add_argument('--suffix', type=str, default="", help='optional suffix to separate n/w')
args = parser.parse_args()



#%%

#def retrain():
    
tf.reset_default_graph()

timestamp = time.strftime("%Y%m%d_%H%M%S");


ontology, colors = read_ontology(args.dataset + '/Ontology.csv')

num_classes = len(ontology)

image_shape=(args.crop_height, args.crop_width) #NB! Now it is height x width !!!

#full_model_name = args.prefix + args.model_name + "_" + str(args.crop_width) + "x" + str(args.crop_height) + args.suffix;
full_model_name = args.prefix + args.model_name + args.suffix;
model_path = args.dataset + '/nets/' + full_model_name;


lr_file = args.dataset + '/nets/' + full_model_name + '_lr.txt';
open(lr_file,'w').write(str(args.learning_rate))


epochs = args.num_epochs
batch_size = args.batch_size


alfa = (127,) #semi-transparent

config = tf.ConfigProto(
   gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95),
   device_count = {'GPU': 1}
)
sess = tf.Session(config = config)




load_net = model_path + '-' + str(args.checkpoint_num)

min_loss_name = 'min_loss.txt'

saver = tf.train.import_meta_graph(load_net + '.meta')


model = tf.get_default_graph()

input_image = model.get_tensor_by_name('image_input:0')
keep_prob = model.get_tensor_by_name('keep_prob:0')
nn_output = model.get_tensor_by_name('layer3_up/BiasAdd:0')
correct_label = model.get_tensor_by_name('correct_label:0')
learning_rate = model.get_tensor_by_name('learning_rate:0')

assert(nn_output.shape[-1] == len(ontology))

train_op, loss, conf_matrix, miou  = optimize(nn_output, correct_label, 
                                    learning_rate, num_classes)



#tf.summary.scalar('val loss', val_loss)
lr_summary = tf.summary.scalar('learning_rate', learning_rate)


sess.run(tf.global_variables_initializer())

print ('Loading ' + load_net)

saver.restore(sess,load_net)

train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'layer3')
#train_op = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss, var_list = train_vars)
#train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss, var_list = train_vars)


train_nn(sess, full_model_name, epochs, batch_size, 
         args.dataset, image_shape, ontology,
         train_op, loss, miou, conf_matrix, saver,
         input_image, correct_label, nn_output, keep_prob, learning_rate, 5000) 




#if __name__ == '__main__':
#    retrain()
           

