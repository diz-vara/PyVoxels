# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:44:33 2019

@author: avarfolomeev
"""

import cv2
import numpy as np
import os
import glob
import sys


def resize_images(input_dir, output_dir, target_shape):
    
    images_dir = os.path.join(input_dir,'images')
    labels_dir = os.path.join(input_dir,'labels')
    
    o_images_dir = os.path.join(output_dir,'images')
    o_labels_dir = os.path.join(output_dir,'labels')
    
    try:
        os.makedirs(o_images_dir)
        os.makedirs(o_labels_dir)
    except:
        pass    
    
    images_list = glob.glob(images_dir+"/*.*g")
    
    target_shape = np.array(target_shape)
    pad_list = []

    for im_file in images_list:
        fname = os.path.split(im_file)[-1]
        fname,ext = os.path.splitext(fname)

        img = cv2.imread(im_file,-1)
        img_shape = np.array(img.shape)
        
        mask = cv2.imread(os.path.join(labels_dir, fname + ".png"),-1)
        
        try:
            assert( (img_shape[:2] == mask.shape[:2]).all())

        
            coeffs = img_shape[:2]/target_shape
            coeff = 1./max(coeffs)
            resized = cv2.resize(img, (0,0), fx=coeff, fy=coeff, 
                                 interpolation=cv2.INTER_LINEAR)
            if (len (resized.shape) < 3):
                resized = np.expand_dims(resized,-1)
    
            pad = target_shape - resized.shape[:2]
            padded = np.pad(resized, ((0,pad[0]), (0,pad[1]),(0,0)), "constant")
    
            pad_file = os.path.join(o_images_dir, fname+ext)
            cv2.imwrite(pad_file, padded)
    
            if (len (resized.shape) < 3):
                resized = np.expand_dims(resized,-1)
    
            resized = cv2.resize(mask, (0,0), fx=coeff, fy=coeff, 
                                 interpolation=cv2.INTER_NEAREST)
            if (len (resized.shape) < 3):
                resized = np.expand_dims(resized,-1)
    
            padded = np.pad(resized, ((0,pad[0]), (0,pad[1]),(0,0)), "constant")
            pad_file = os.path.join(o_labels_dir, fname+".png")
            cv2.imwrite(pad_file, padded)
    
    
            pad_list.append( (fname, pad[0], pad[1]))
        except:
            print (fname)
            print (sys.exc_info())
            pass
        
    return np.array(pad_list)
        
    