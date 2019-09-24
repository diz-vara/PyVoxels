# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 17:30:16 2019

@author: avarfolomeev
"""

import os
import glob
import scipy.misc
import numpy as np
import csv
import cv2
import argparse


def read_recode_table(fname, delimiter = ','):

    conv = []
    with open (fname) as csvfile:
        t_reader = csv.reader(csvfile, delimiter = delimiter)
        next(t_reader,None)
        for row in t_reader:
            in2out = tuple((int(row[0]), int(row[3])))
            conv.append(in2out)
    
    return np.array(conv)

#%%
    
def re_code_labels(base_dir, in_dir, out_dir,table_csv):

    re_code = read_recode_table(table_csv)
    
    inp = re_code[:,0]
    out = re_code[:,1]    
    
    ncolors = len(out)

    for i in range (len(inp)):
        if i != inp[i]:
            print( 'wrong input lable',inp[i], ' at position', i)        
            exit(1)
    
    in_dir = os.path.join(base_dir, in_dir)
    out_dir = os.path.join(base_dir, out_dir)

    try:
        os.makedirs(out_dir)
    except:
        pass


    print('Loading input labels from ' + in_dir)
    print('Saving output labels to ' + out_dir)
    
    
    im_files = sorted(os.listdir(in_dir))
    cnt = 0
    end_string = ' from ' + str(len(im_files))
    

    for label_file in im_files:
        print(str(cnt) + end_string)
        cnt = cnt+1
        label_in = cv2.imread(os.path.join(in_dir, label_file),-1)
        if (len(label_in.shape) > 2):  #if RGB, use only R
            label_in = label_in[:,:,2]
        label_in[label_in >= ncolors] =  0
        label_out = out[label_in]
        cv2.imwrite(os.path.join(out_dir, label_file), label_out)
                    