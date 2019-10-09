# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 17:23:49 2018

@author: avarfolomeev
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

from read_ontology import read_ontology

import argparse 

#%%
def renum_objects(base_dir, in_dir, out_dir, ontology, min_size = 100):
    """Load road masks from file.
    Images are RGB, with:
        (255,0,255) for road
        (255,0,0) for not_road
    """
    in_dir = os.path.join(base_dir, in_dir)
    out_dir = os.path.join(base_dir, out_dir)

    try:
        os.makedirs(out_dir)
    except:
        pass


    print('Loading object masks from ' + in_dir)

    
    has_objects = np.array([label.has_objects for label in ontology]).astype(np.uint8)
    classes = np.array([label.id for label in ontology]).astype(np.uint8)
    num_classes = len(classes)

    im_files = sorted(os.listdir(in_dir))
    cnt = 0
    end_string = ' from ' + str(len(im_files))
    
    
    
    for label_file in im_files:
        label_in = (plt.imread(os.path.join(in_dir, label_file))*255.).astype(np.uint8)
        _classes = label_in[:,:,0]
        _objects = label_in[:,:,1]
        
        _objectness = np.not_equal(_objects,0)
        
        new_objects = np.zeros_like(_objects)
        
        num_instances=0
        for i in range(num_classes):
            if (has_objects[i]):
                _class = classes[i]
                _class_objects_mask = np.logical_and(np.equal(_class,_classes),_objectness)
                _class_objects_mask = np.uint8(_class_objects_mask)
                _class_objects = np.multiply(_objects, _class_objects_mask)
                #here I've got All objects, I want filter out small ones
                # and re-number !!!
                _new_class_objects = np.zeros_like(_class_objects)
                new_idx = 1;
                for obj in np.arange (1,np.max(_class_objects)+1):
                    _current_obj = np.int8(np.equal(obj,_class_objects))
                    _curr_size = np.sum(np.int32(_current_obj))
                    #print (_class, new_idx, _curr_size)
                    if ( _curr_size >= min_size):
                        _new_class_objects = np.add(_new_class_objects,_current_obj * new_idx);
                        new_idx += 1
                
                _new_objects_mask = np.uint8(np.not_equal(_new_class_objects,0))
                _new_class_objects = np.add(_new_class_objects, num_instances)
                        
                _new_class_objects = np.multiply(_new_class_objects, _new_objects_mask) #remove zeros
                
                
                new_objects = np.add(new_objects, _new_class_objects)
                num_instances = np.max(new_objects)
                

        label_in[:,:,1] = new_objects

        plt.imsave(os.path.join(out_dir, label_file),label_in)
        cnt += 1
        print (num_instances, label_file, cnt, end_string)
    




#%%
        
#%%%
if (__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type = str, default = "", help = "Base directory")
    parser.add_argument("dir_in", type = str, default = "", help = "Input directory (color labels)")
    parser.add_argument("dir_out", type = str, default = "", help = "Output directory (grey labels)")
    parser.add_argument("ontology", type = str, default = "", help = "Ontology.csv (in F8 format")
    parser.add_argument("--min_size", type = int, default = 100, help = "minimal object size")
    args = parser.parse_args()
        
    ont = read_ontology(args.ontology)
    print ("renum objects from ", args.dir_in, " to " , args.dir_out)
    renum_objects(args.base_dir, args.dir_in, args.dir_out, ont[0], 
                  min_size = args.min_size)
        