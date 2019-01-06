# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 22:07:52 2019

@author: diz
"""

import glob
import numpy as np


def write_list(fname, _list):
    with open(fname,'w') as file:
        for l in _list:
            str = l[0] + ', ' + l[1] + '\n'
            
            file.write(str)
    file.close()    

def prepare_data_lists(data_dir, percents=(75,15,10)):
    assert sum(percents)==100, "Sum of percents is not 100"
    
    images = glob.glob(data_dir+'/*.jpg')
    labels = glob.glob(data_dir+'/*.png')
    
    i2l = [s.replace('.jpg', '.png') for s in images]

    all_files = np.array([z for z in zip(images, labels)])

    assert len(images) == len(labels), \
        "lists are different: %r images and %r labels" % (len(images), len(labels))
     
    if (i2l != labels):
        i2l_ = np.array(i2l)
        labels_ = np.array(labels)
        idx = (i2l_ != labels_)
        print(all_files[idx])
    
        
    assert (i2l == labels), "image and label lists are different: "
    
    
    n_units = len(images)
    n_test = (n_units * percents[2]) // 100
    if (n_test < 1):
        n_test = 1

    n_val = (n_units * percents[1]) // 100
    if (n_val < 1):
        n_val = 1;
    
    #print(n_units, n_units-(n_test+n_val), n_val, n_test)
    
    idx = np.arange(n_units).astype(int)
    np.random.shuffle(idx)
    test = idx[:n_test]
    val = idx[n_test:n_test+n_val]
    train = idx[n_test+n_val:]

    write_list(data_dir + "/train.txt", all_files[train])
    write_list(data_dir + "/trainval.txt", all_files[val])
    write_list(data_dir + "/val.txt", all_files[test])
    
    

    return #all_files, train, val
    
    
    