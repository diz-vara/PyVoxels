# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 09:19:33 2018

@author: avarfolomeev
"""

import numpy as np
import cv2


def filter_cloud(cloud, limits=None):
    if (limits == None):
        limits = np.array([[0.1, 30], [-10,10], [-3,30]])
    filt = ( cloud[:,0] > limits[0,0]) \
            & (cloud[:,1] > limits[1,0]) \
            & (cloud[:,1] < limits[1,1]) \
            & (cloud[:,2] > limits[2,0]) \
            & (cloud[:,2] < limits[2,1]) \
            & (cloud[:,0] < limits[0,1])
    idx = np.arange(len(cloud))
    return cloud[filt,:], idx[filt]
        

def filter_pixels(pts, shape = (1920,1200)):
    pts=pts.round()
    pts_idx= (pts[:,0] >= 0) & (pts[:,0]< shape[0]) & (pts[:,1] >= 0) & (pts[:,1] < shape[1])
    return pts_idx

def project_cloud(cloud, image, calibration):    
    
    pts, jac = cv2.projectPoints(cloud, 
                                 calibration['rot'], 
                                 calibration['t'], 
                                 calibration['mtx'], 
                                 calibration['dist'])
    pts = pts[:,0,:]
    pts_idx = filter_pixels(pts,image.shape)
    ff_cloud = cloud[pts_idx,:]
    f_pts = np.round(pts[pts_idx]).astype(int)
    colors = image[f_pts[:,1],f_pts[:,0]]
    idx = idx[pts_idx]
    return ff_cloud, colors, idx
    

def l_project_cloud(cloud, image, calibration):    
   
    pts, jac = cv2.projectPoints(cloud, 
                                 calibration['rot'], 
                                 calibration['t'], 
                                 calibration['mtx'], 
                                 calibration['dist'])
    pts = pts[:,0,:]
    pts_idx = filter_pixels(pts,image.shape)
    ff_cloud = cloud[pts_idx,:]
    f_pts = np.round(pts[pts_idx]).astype(int)
    colors = image[f_pts[:,1],f_pts[:,0]]
    idx = np.arange(len(cloud))[pts_idx]
    return ff_cloud, colors, idx
    
    
