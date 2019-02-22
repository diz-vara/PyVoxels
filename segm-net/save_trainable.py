#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 23:01:52 2019

@author: avarfolomeev
"""

 
import tensorflow as tf


tf.reset_default_graph()

data_dir = '/media/avarfolomeev/storage/Data/Segmentation/data'
runs_dir = '/media/avarfolomeev/storage/Data/Segmentation/vox_segm/runs'

config = tf.ConfigProto(
   gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
   device_count = {'GPU': 1}
)
sess = tf.Session(config = config)


load_net = '/media/avarfolomeev/storage/Data/Segmentation/vox_segm/vox-net-lp-673'

saver = tf.train.import_meta_graph(load_net + '.meta')
saver.restore(sess,load_net)

#saver_m = tf.train.Saver(tf.model_variables())
#saver.save(sess,load_net + "_model")


saver_m = tf.train.Saver(tf.trainable_variables())
saver.save(sess,load_net + "_trainable")

