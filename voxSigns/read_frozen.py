#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 17:46:05 2019

@author: avarfolomeev
"""

import tensorflow as tf



def read_frozen(frozen_path):
    new_graph = tf.Graph()
    
    with new_graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_path, 'rb') as fid:
            graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name = '')
    

    return new_graph
