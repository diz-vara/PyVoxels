# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 14:12:16 2018

@author: avarfolomeev
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 

def read_cloud_csv(cloud_num, base_dir = 'E:\\Data\\Voxels\\London-cal1\\selected\\'):

    fname = base_dir + '\\{:06}.csv'.format(cloud_num)
    with open (fname) as csvfile:
        pclreader = csv.reader(csvfile, delimiter = ',')
        points = []
        for row in pclreader:
            pnt = np.array([float(p) for p in row])
            points.append(pnt)
        
    cloud = np.array(points)        
    return cloud
    
    
        
def scatt3d(ax, cloud, clear = False, color = 'g', marker = 'o', size=3):
    cloud = np.array(cloud)
    if (clear):
        ax.cla()    
    if (len(cloud.shape) < 2):
        cloud = np.expand_dims(cloud,0)
    ax.scatter3D(cloud[:,0], cloud[:,1], cloud[:,2], 
                 c = color, 
                 marker=marker,
                 s=size)


def scatt2d(ax, cloud, clear = False, color = 'g', marker = 'o', size=3):
    cloud = np.array(cloud)
    if (clear):
        ax.cla()    
    if (len(cloud.shape) < 2):
        cloud = np.expand_dims(cloud,0)
    ax.scatter(cloud[:,0], cloud[:,1], 
                 c = color, 
                 marker=marker,
                 s=size)

