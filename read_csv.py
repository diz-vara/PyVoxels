# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 14:12:16 2018

@author: avarfolomeev
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 
import cv2

def read_cloud_csv(cloud_num, base_dir = 'E:\\Data\\Voxels\\London-cal1\\selected\\'):

    fname = base_dir + '\\{:06}.csv'.format(cloud_num)
    with open (fname) as csvfile:
        pclreader = csv.reader(csvfile, delimiter = ',')
        points = []
        for row in pclreader:
            pnt = np.array([float(p) for p in row])
            points.append(pnt)
        
    cloud = np.array(points)     
    cloud[:,1] = cloud[:,1] * -1
    return cloud
    


def read_full_cloud_csv(cloud_num, base_dir = 'E:\\Data\\Voxels\\London-cal1\\London-cal1_drive_0015_sync\\velodyne_points\\csv'):

    fname = base_dir + '\\{:010}.csv'.format(cloud_num)
    with open (fname) as csvfile:
        pclreader = csv.reader(csvfile, delimiter = ' ')
        points = []
        for row in pclreader:
            pnt = np.array([float(p) for p in row])
            points.append(pnt)
        
    cloud = np.array(points)     
    cloud = cloud[:,[1,3,2]]
    cloud[:,1] = cloud[:,1] * -1
    return cloud

    
        
def scatt3d(ax, cloud, clear = False, color = 'g', marker = 'o', size=25):
    cloud = np.array(cloud)
    if (clear):
        ax.cla()    
    if (len(cloud.shape) < 2):
        cloud = np.expand_dims(cloud,0)
    ax.scatter3D(cloud[:,0], cloud[:,1], cloud[:,2], 
                 c = color, 
                 marker=marker,
                 s=size,edgecolors='face')


def scatt2d(ax, cloud, clear = False, color = 'g', marker = 'o', size=25):
    cloud = np.array(cloud)
    if (clear):
        ax.cla()    
    if (len(cloud.shape) < 2):
        cloud = np.expand_dims(cloud,0)
    ax.scatter(cloud[:,0], cloud[:,1], 
                 c = color, 
                 marker=marker,
                 s=size,edgecolors='face')

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img
    