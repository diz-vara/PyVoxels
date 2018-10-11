# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 09:17:04 2017

@author: Anton Varfolomeev
"""

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

#real-world coordinates of the calibration pattern
def calcCorners(nx,ny, sqSide = 40):
    objCorners = []
    for row in range(ny):
        for col in range (nx):
            x = sqSide*(col+1);
            y = sqSide*(row+1);
            objCorners.append([x,y,0.0])
    return np.array(objCorners, dtype=np.float32)      

#%%
def calibrate(cal_dir = './camera_cal', nx=9, ny=6, table = None):
    
    #we can skip it here, but for real calibration FAST_CHECK is important!!
    flagCorners = cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH 
    #flagCorners = flagCorners | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS
    #enable additional coefficients
    flagCalib = 0
    #flagCalib = flagCalib | cv2.CALIB_RATIONAL_MODEL
    #flagCalib = flagCalib | cv2.CALIB_THIN_PRISM_MODEL
    flagCalib = flagCalib | cv2.CALIB_TILTED_MODEL
    
    nDist = 8
    
    if (flagCalib & cv2.CALIB_THIN_PRISM_MODEL):
        nDist = 12
        
    print('using', nDist, 'distortion coeffs')
            
    imgPoints = []
    objPoints = []        
    files = []
    #read images and try to find corneres in each of them        
    for entry in os.scandir(cal_dir):
        if entry.is_file():
            img = cv2.imread(entry.path)
            if (not table is None):
                img = cv2.LUT(img, table)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            shape = img.shape[0:2]
            _nx = nx
            _ny = ny
            ret, corners = cv2.findChessboardCorners(img, (_nx,_ny), flagCorners)
            print(entry.name, ret)
            #plt.imshow(img, cmap = 'gray')
            #if (not ret):
            #    _ny = ny - 1
            #    ret, corners = cv2.findChessboardCorners(img, (_nx,_ny), flagCorners)
            #    print((_nx,_ny),entry.name,  ret)
                
            #if (not ret):
            #    _nx = nx - 1
            #    _ny = ny
            #    ret, corners = cv2.findChessboardCorners(img, (_nx,_ny), flagCorners)
            #    print((_nx,_ny),entry.name, ret)

            if (ret):
                #corners found - add object and image coordinates
                print("adding ", entry.name, (_nx,_ny))
                objPoints.append(calcCorners(_nx,_ny))
                imgPoints.append(corners)
                files.append(entry.path)
                
    #return files, np.array(imgPoints)

    #objPoints = np.array(objPoints, dtype=np.float32)            
    #imgPoints = np.array(imgPoints)            
    #all corners found - build calibration matrix and 
    #calculate distortion coeffs
    dcf = np.zeros((1,nDist), np.float64)
    mtx = np.zeros((3,3), np.float64)
    ret, mtx, dist, rv, tv = cv2.calibrateCamera(objPoints, imgPoints, shape, 
                                                 mtx, distCoeffs=dcf, 
                                                 flags = flagCalib)
    return ret, mtx, dist

#%%    
def build_lut (invgamma):
    table = np.array([ ((i/255.0) ** invgamma) * 255 for i in np.arange(0,256)]).astype("uint8")    
    return table
   
#%%
def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))

    