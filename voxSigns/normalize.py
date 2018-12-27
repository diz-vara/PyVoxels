#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 12:15:51 2018

@author: avarfolomeev
"""

import numpy as np



#per-channel normalization
def normalizeImageC(img):
        imf = np.float32(img); #cv2.cvtColor(img,cv2.COLOR_RGB2Lab));
        mn = np.mean(np.mean(imf[5:27][5:27],0),0);
        s = np.max(np.max(imf[5:27][5:27],0),0) - np.min(np.min(imf[5:27][5:27],0),0) + 10;
        for chan in range(img.shape[2]):
            imf[:,:,chan] = imf[:,:,chan] - mn[chan];
            imf[:,:,chan] = imf[:,:,chan] / s[chan];
        return imf;


#global normalization
def normalizeImageG(img):
        imf = np.float32(img);
        imf = imf - np.mean(imf[5:27][5:27])
        s = np.max(imf[5:27][5:27]) - np.min(imf[5:27][5:27])
        imf = imf / (s + 1)
        return imf;

#global normalization
def normalizeImage0(img):
        imf = np.float32(img);
        imf = (imf - 128) / 128.
        return imf;
    
#%%    

def normalizeImageList(imgList, mode = 'G'):
    if (mode == 'C'):
        print('Channel-wize normalization');
        out = np.array([normalizeImageC(img) for img in imgList]);
    elif (mode == 'G'):
        print('Global normalization');
        out = np.array([normalizeImageG(img) for img in imgList]);
    elif (mode == '0'):
        print('0- normalization (x-128)/128');
        out = np.array([normalizeImage0(img) for img in imgList]);
    return out;
    
    
