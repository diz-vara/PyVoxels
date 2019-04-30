#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 12:02:07 2018

@author: avarfolomeev
"""

import os
import glob
import scipy.misc
import numpy as np

from normalize import *
from augment import *
from splitIndicies import *

def read_texts(root_dir):
    
    filenames =  [f for f in glob.iglob(root_dir + '/**/*.*g', recursive=True)]
        

    units = []
    for f in filenames:
        
        img = scipy.misc.imread(f, mode = 'RGB')
        (w,h,c) = img.shape

        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        
        """
        if ( abs (w-h) > 12) :
            continue;
            
        if ( w != h):
            w = min(w,h)
            img = img[:w,:w,:]
        """    
        folder = os.path.split(f)[0];
        folder, thin_class = os.path.split(folder)
        thick_class = thin_class.split('_')[0]

        #thick_class = thin_class
        #while (folder != root_dir and folder != '/'):
        #    folder, thick_class = os.path.split(folder)
            
        units.append((img, thin_class, thick_class))

    return units

r = read_texts(root_dir)
images = [_r[0] for _r in r]
t_thin = np.array([_r[1] for _r in r])
t_thick = np.array([_r[2] for _r in r])

classes_thin = np.unique(t_thin)
classes_thick = np.unique(t_thick)

#%%
#r = read_signs(root_dir)

nt_thin = np.zeros(len(r))

for c in range(len(classes_thin)):
    nt_thin[t_thin == classes_thin[c]] = c
    
nt_thick = np.zeros(len(r))

for c in range(len(classes_thick)):
    nt_thick[t_thick == classes_thick[c]] = c
    
#%%
    

#count number of each class examples
#and store the index of the last one

nt_ = nt_thick
classes_ = classes_thick

y_all = nt_.astype(int)   
n_classes = len(classes_)
    
nSamples = len(y_all)    

X_all = np.zeros((nSamples,24,36),dtype='uint8')

for i in range(nSamples):
    X_all[i] = scipy.misc.imresize(images[i],(24,36))

#NB! Change to thick here!!!!
    
classIndicies = [np.where(y_all == i)[0] for i in range(n_classes)]
classCounts = [np.size(array) for array in classIndicies]

v,t = splitIndicies(classIndicies,12)

Xgn_all = normalizeImageList(X_all,'0')

X_train = Xgn_all[t]
X_val = Xgn_all[v]
y_train = y_all[t]
y_val = y_all[v]


(Xgn_t, Yg_t) = augmentImageList(X_train,y_train,8000)



#%%
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
get_ipython().magic('matplotlib inline')
import matplotlib.gridspec as gridspec

#display example of each class and show number of samples
cols = 5
figsize = (10, 10)

gs = gridspec.GridSpec((n_classes-2) // cols + 1, cols)

fig1 = plt.figure(num=1, figsize=figsize)
ax = []

exShape = list(X_train.shape);
exShape[0] = n_classes;
examples = np.empty(exShape, dtype=np.uint8)
for i in range(1,n_classes):
    row = ((i-1) // cols)
    col = (i-1) % cols
    ax.append(fig1.add_subplot(gs[row, col]))
    ax[-1].set_title('%s\n(%d) N=%d' % (classes_[i] ,i+200,   classCounts[i]))
    #example
    img = X_all[classIndicies[i][0]]
    #rescale to make dark images visible
    cf = np.int(255/np.max(img)) 
    examples[i] = img*cf;
    ax[-1].imshow(img*cf, cmap='gray')
    ax[-1].axis('off')
    
#%%
    
    
    