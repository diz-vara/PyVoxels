#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 11:32:39 2018

@author: undead
"""

import tensorflow as tf
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, image_shape)
  return image_resized

# A vector of filenames.
#filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])




def get_images(file_list, batch_size = 10):
        filename_queue = tf.train.string_input_producer(file_list, shuffle = False)
        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)
        image = tf.image.decode_jpeg(value, channels = 3)
        return image
    
        
dataset = tf.data.Dataset.from_tensor_slices((ll))
dataset = dataset.map(_parse_function)
    
iterator = dataset.make_one_shot_iterator()

next_element = iterator.get_next()

