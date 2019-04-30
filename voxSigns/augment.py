#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 14:19:18 2018

@author: avarfolomeev
"""


import numpy as np
import math
import cv2

def getMotionKernel(size):
    kernel = np.zeros((size, size));
    kernel[int((size-1)/2), :] = np.ones(size)/size;
    return kernel
    
motion_kern3 = getMotionKernel(3);
motion_kern5 = getMotionKernel(5);    
    
def getPerspMatrix(x, y, z, size):
    w, h = size;
    half_w = w/2.;
    half_h = h/2.;

    
    rx = math.radians(x);
    ry = math.radians(y);
    rz = math.radians(z);
    
    cos_x = math.cos(rx);
    sin_x = math.sin(rx);
    cos_y = math.cos(ry);
    sin_y = math.sin(ry);
    cos_z = math.cos(rz);
    sin_z = math.sin(rz);
 
     # Rotation matrix:
    # | cos(y)*cos(z)                       -cos(y)*sin(z)                     sin(y)         0 |
    # | cos(x)*sin(z)+cos(z)*sin(x)*sin(y)  cos(x)*cos(z)-sin(x)*sin(y)*sin(z) -cos(y)*sin(y) 0 |
    # | sin(x)*sin(z)-cos(x)*sin(y)*sin(z)  sin(x)*sin(z)+cos(x)*sin(y)*sin(z) cos(x)*cos(y)  0 |
    # | 0                                   0                                  0              1 |

    R = np.float32(
        [
            [cos_y * cos_z,  cos_x * sin_z + cos_z * sin_y * sin_x],
            [-cos_y * sin_z, cos_z * cos_x - sin_z * sin_y * sin_x],
            [sin_y,          cos_y * sin_x],
        ]
    );

    center = np.float32([half_h, half_w]);
    offset = np.float32(
        [
            [-half_w, -half_h],
            [ half_w, -half_h],
            [ half_w,  half_h],
            [-half_w,  half_h],
        ]
    );

    points_z = np.dot(offset, R[2]);
    dev_z = np.vstack([w/(w + points_z), h/(h + points_z)]);

    new_points = np.dot(offset, R[:2].T) * dev_z.T + center;
    in_pt = np.float32([[0, 0], [w, 0], [w, h], [0, h]]);

    transform = cv2.getPerspectiveTransform(in_pt, new_points);
    return transform;



def transformImg(img, x=0, y=0, z=0, scale = 1):
    size = img.shape[2::-1]
    
    M = getPerspMatrix(x, y, z, size)
    if scale != 1:
        S = np.eye(3);
        S[0,0] = S[1,1] = scale;
        S[0,2] = size[0]/2 * (1-scale);
        S[1,2] = size[1]/2 * (1-scale);
        M = np.matmul(S,M);

    result = cv2.warpPerspective(img, M, size, borderMode=cv2.BORDER_REFLECT)

    return result

    
                             
#replicate img N times
def augmentImage(img, N:int):
    
    out = [img];

    rangeX = [  0, 2];
    rangeY = [-5, 5];
    rangeZ = [-5, 5];
    rangeS = [0.8, 1.2]
    rangeI = [-0.3, 0.3];


    for i in range(N-1):
        x = np.random.uniform(rangeX[0], rangeX[1]);
        y = np.random.uniform(rangeY[0], rangeY[1]);
        z = np.random.uniform(rangeZ[0], rangeZ[1]);
        scale = np.random.uniform(rangeS[0], rangeS[1]);
        motion = 0; #np.random.uniform();
        if motion > 0.8:
            tmp = cv2.filter2D(img,-1,motion_kern5);
        elif motion > 0.5 :
            tmp = cv2.filter2D(img,-1,motion_kern3);
        else:
            tmp = img;
        intens = np.random.uniform(rangeI[0], rangeI[1]);
        tmp = np.clip(tmp+intens,-0.5,0.5);
        out.append(transformImg(tmp,x,y,z,scale));
    return out;


#%%    
#build new list of N images    
def augmentImgClass(imgList, outOrN ):
    shape = list(imgList.shape);
    inputLen = shape[0];
    if (type(outOrN) == int):
        outLen = outOrN;
        shape[0] = outLen #to form output array
        out = np.empty(shape, np.float32)
    elif (type(outOrN)==np.ndarray):
        out = outOrN;
        outLen = out.shape[0];
    else:
        print("invalid second argument")
        return np.empty(0)
        
        

    k = 0
    l = 0
    for  img in imgList:
        cf = np.int((outLen-k)/(inputLen-l)) + 1;
        if (cf > 1):
            newImages = augmentImage(img, cf);
            l = l+1;
            for imNew in newImages:
                #print ( img.shape, imNew.shape)
                if (k < outLen):
                    out[k]=imNew;
                k = k+1;
        else:
            if (k < outLen):
                out[k] = img;
            k = k+1;
    #print (l,k,cf)
    return out;
        
                                
#%%

def augmentImageList(X,Y,targetCount):
    n_classes = max(Y) + 1
    indicies = [np.where(Y == i)[0] for i in range(n_classes)]
    totalLen = targetCount * n_classes;
    targetXShape = list(X.shape);
    targetXShape[0] = totalLen; 
    
    targetX = np.empty(targetXShape,dtype = np.float32);
    targetY = np.empty(targetXShape[0], dtype = np.uint8);
                     
    for signClass in range(n_classes):
        print("filling class ", signClass);
        inputImages = X[indicies[signClass]];
        augmentImgClass(inputImages, targetX[signClass*targetCount:(signClass+1)*targetCount]);
        targetY[signClass*targetCount:(signClass+1)*targetCount] = signClass;

    idx = np.arange(totalLen);
    np.random.shuffle(idx);
        #shuffle
    targetY = targetY[idx];
    targetX = targetX[idx];
    
    return (targetX, targetY);

