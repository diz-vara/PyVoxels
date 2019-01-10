# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 17:23:49 2018

@author: avarfolomeev
"""
import glob
import os
import matplotlib.pyplot as plt
import numpy as np


#import labels_xu as lbl_xu
#labels = lbl_xu.labels
#   colors = np.array([label.color for label in labels]).astype(np.uint8)

def label2color(base_dir, in_dir, out_dir, colors):
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

    
    

    im_files = sorted(os.listdir(in_dir))
    cnt = 0
    end_string = ' from ' + str(len(im_files))
    
    
    
    for label_file in im_files:
        print(str(cnt) + end_string)
        cnt = cnt+1
        label_in = (plt.imread(os.path.join(in_dir, label_file))*255.).astype(np.uint8)
        if (len(label_in.shape) > 1):
            label_in = label_in[:,:,0]
        colors_out = colors[label_in]
        plt.imsave(os.path.join(out_dir, label_file),colors_out)
    
    
def colors2label(base_dir, in_dir, out_dir, colors):
    
    in_dir = os.path.join(base_dir, in_dir)
    out_dir = os.path.join(base_dir, out_dir)

    try:
        os.makedirs(out_dir)
    except:
        pass


    print('Loading masks from ' + in_dir)

    
    

    im_files = sorted(os.listdir(in_dir))
    cnt = 0
    end_string = ' from ' + str(len(im_files))
    
    
    
    
    
    for label_file in im_files:
        print(str(cnt) + end_string)
        cnt = cnt+1
        colors_in = cv2.imread(os.path.join(in_dir, label_file))
        colors_in = cv2.cvtColor(colors_in,cv2.COLOR_RGB2BGR)
        
        colors_blur = cv2.GaussianBlur(colors_in,(3,3),1);
        
        
        
        sh = colors_in.shape

        label = np.zeros ((sh[0], sh[1]), dtype = np.uint8)

        
        for idx in range(len(colors)):
            color = colors[idx]
            s = (colors_blur == color).all(axis=2)
            label[s] = idx            
        
        cv2.imwrite(os.path.join(out_dir, label_file), label)
        