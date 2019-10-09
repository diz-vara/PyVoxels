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


#import labels_xu as lbl_xu
#labels = lbl_xu.labels
#   colors = np.array([label.color for label in labels]).astype(np.uint8)

def labels2colors(base_dir, in_dir, out_dir, ontology, channel = 0):
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


    print('Loading road masks from ' + in_dir)

    
    colors = np.array([label.color for label in ontology]).astype(np.uint8)
    labels = np.array([label.id for label in ontology]).astype(np.uint8)
    num_colors = len(colors)
    

    im_files = sorted(os.listdir(in_dir))
    cnt = 0
    end_string = ' from ' + str(len(im_files)) + " "
    
    
    
    for label_file in im_files:
        cnt = cnt+1
        label_in = (plt.imread(os.path.join(in_dir, label_file))*255.).astype(np.uint8)
        if (len(label_in.shape) > 2):
            label_in = label_in[:,:,channel]

        sh = label_in.shape
        max_label = np.max(label_in)
        print(str(cnt) + end_string, max_label)

        #label = np.ones ((sh[0], sh[1]), dtype = np.uint8)*255
        colors_out = np.zeros ((sh[0], sh[1],3), dtype = np.uint8)

        for idx in np.arange(1,max_label+1):
            #label = labels[idx % num_colors]
            s = (label_in == idx) #.all(axis=2)
            colors_out[s] = colors[idx % num_colors]
            #colors_out[s] += (idx // num_colors) * 2            

        plt.imsave(os.path.join(out_dir, label_file),colors_out)
    
    

        
#%%%
if (__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type = str, default = "", help = "Base directory")
    parser.add_argument("dir_in", type = str, default = "", help = "Input directory (color labels)")
    parser.add_argument("dir_out", type = str, default = "", help = "Output directory (grey labels)")
    parser.add_argument("ontology", type = str, default = "", help = "Ontology.csv (in F8 format")
    parser.add_argument("--channel", type = int, default = 0, help = "Channel to decode")
    args = parser.parse_args()
        
    ont = read_ontology(args.ontology)
    print ("convert from ", args.dir_in, " to " , args.dir_out)
    labels2colors(args.base_dir, args.dir_in, args.dir_out, ont[0], args.channel)
        