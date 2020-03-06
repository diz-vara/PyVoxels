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
import glob

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
  # cv2.CALIB_TILTED_MODEL
  # cv2.CALIB_RATIONAL_MODEL
  # cv2.CALIB_THIN_PRISM_MODEL

def calibrate(cal_dir = './camera_cal', nx=9, ny=6, nSamples = -1, step = 1, 
              table = None,
              model=cv2.CALIB_TILTED_MODEL,
              scale  = 1.,
              draw_fig=None, nCols = 5, sqSide = 40.):
    
    
    if (nSamples > 0 and nCols > nSamples):
        nCols = nSamples
    
    #we can skip it here, but for real calibration FAST_CHECK is important!!
    flagCorners = cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH 
    #enable additional coefficients
    flagCalib = 0
    flagCalib = flagCalib | model 
    
    nDist = 8
    
    if (flagCalib & cv2.CALIB_THIN_PRISM_MODEL):
        nDist = 12
        
    print('using', nDist, 'distortion coeffs')
            
    imgPoints = []
    objPoints = []        
    files = []

    file_list = glob.glob(cal_dir+'/*.jpg')
    
    
    
    start = int(np.random.uniform(step))
    
    file_list = file_list[start::step]
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    
    np.random.shuffle(file_list)
    if (nSamples > 0):
        file_list = file_list[:nSamples]
    else:
        nSamples = len (file_list)
        
    nRows = nSamples // nCols
    if ( nSamples % nCols):
        nRows += 1
        
    ax = ()  
    if (draw_fig):
        draw_fig.clf()
        ax = draw_fig.subplots(nRows, nCols) 

    #read images and try to find corneres in each of them        
    cnt = 0
    for entry in file_list:
        img = cv2.imread(entry)
        if (not table is None):
            img = cv2.LUT(img, table)
        grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        
        shape = img.shape[1::-1]
        ret, corners = cv2.findChessboardCorners(grey, (nx, ny), flagCorners)
        if (ret):
            cv2.cornerSubPix(grey,corners,(nx,ny), (-1,-1),term)

        
        if (scale <= 0):
            scale = 1.

                             
        name = os.path.split(entry)[1]
        print(name, ret)

        row = cnt // nCols
        col = cnt % nCols
        if (ret):
            #corners found - add object and image coordinates
            print("adding ", name, shape)
            objPoints.append(calcCorners(nx,ny, sqSide))
            imgPoints.append(corners)
            files.append(entry)
            if draw_fig is not None:
                cv2.drawChessboardCorners(img,(nx,ny), corners, ret)
                if (nRows > 1):
                    ax[row,col].imshow(img )
                else:
                    ax[col].imshow(img )
                cv2.imwrite("t/"+name, img);
                #plt.waitforbuttonpress()
        cnt += 1
                
    #return files, np.array(imgPoints)

    #objPoints = np.array(objPoints, dtype=np.float32)            
    #imgPoints = np.array(imgPoints)            
    #all corners found - build calibration matrix and 
    #calculate distortion coeffs
    dcf = np.zeros((1,nDist), np.float64)
    
    ret, mtx, dist, rv, tv = cv2.calibrateCamera(objPoints, np.array(imgPoints)*scale, shape, 
                                                 np.eye(3), distCoeffs=dcf,
                                                 flags = flagCalib)
    print('ret=',ret, ', fx=', mtx[0,0], ", fy=", mtx[1,1])
    return ret, mtx, dist #, objPoints, imgPoints, shape, flagCalib

#%%    
def build_lut (invgamma):
    table = np.array([ ((i/255.0) ** invgamma) * 255 for i in np.arange(0,256)]).astype("uint8")    
    return table
   
#%%
def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))

#%%
def show_corners(image_name, ax, pattern_size=(11,11)):
    img = cv2.imread(image_name)
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    flagCorners = cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH 
    ret, corners = cv2.findChessboardCorners(img, pattern_size, flagCorners)
    cv2.drawChessboardCorners(img,(11,11),corners,ret)
    ax.imshow(img)
    return (ret,corners)
#%%
def label_corners(image_name, ax=None, pattern_size=(11,11)):
    img = cv2.imread(image_name)
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    flagCorners = cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH 
    ret, corners = cv2.findChessboardCorners(grey, pattern_size, flagCorners)
    if (ret):
        corners = corners[:,0,:]
        for c in corners.astype(int):
            img[c[1],c[0]]=(0,0,255)
        if (not ax is None):
            img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img1)
        return img
        

#%%
def label_and_save_corners(image_dir, ax=None, pattern_size=(11,11)):
    for entry in os.scandir(image_dir):
        if entry.is_file():
            filename = entry.path
            print(filename)
            img = label_corners(filename,ax,pattern_size)
            if (not img is None):
                outname = filename.replace('jpg', 'png', -1)
                print ("OK ", outname)
                cv2.imwrite( outname, img  )
            else:
                print("fail")
    