# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:04:21 2019

@author: avarfolomeev
"""
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 
import cv2
import matplotlib.pyplot as plt

from rotationMatrix2Euler import eulerAnglesToRotationMatrix


def scatt3d(ax, cloud, clear = False, color = 'g', marker = 'o', size=25):
    cloud = np.array(cloud)
    if (clear):
        ax.cla()    
    if (len(cloud.shape) < 2):
        cloud = np.expand_dims(cloud,0)
    
        
    if (color is None):
        color = np.arange(cloud.shape[0])
    ax.scatter3D(cloud[:,0], cloud[:,1], cloud[:,2], 
                 c = color, 
                 marker=marker,
                 s=size,edgecolors='face')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #right_border = np.ceil( max(cloud[:,1]) * 5.) / 5.
    #left_border = np.floor( min(cloud[:,1]) * 5.) / 5.
    #ax.set_ylim(left_border, right_border)
    


def scatt2d(ax, cloud, clear = False, color = None, marker = 'o', size=25):
    cloud = np.array(cloud)
    if (clear):
        ax.cla()    
    if (len(cloud.shape) < 2):
        cloud = np.expand_dims(cloud,0)

    #color sequence    
    if (color is None):
        color = np.arange(cloud.shape[0])
    ax.scatter(cloud[:,0], cloud[:,1], 
                 c = color, 
                 marker=marker,
                 s=size,edgecolors='face')
    
    
def plot_rot(ax, rot):
    xyz = np.eye(3)
    rm = eulerAnglesToRotationMatrix(rot)
    xyz = xyz*rm
    ax.plot([0,xyz[0,0]], [0,xyz[0,1]],[0,xyz[0,2]], 'r')
    ax.plot([0,xyz[1,0]], [0,xyz[1,1]],[0,xyz[1,2]], 'g')
    ax.plot([0,xyz[2,0]], [0,xyz[2,1]],[0,xyz[2,2]], 'b')

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

