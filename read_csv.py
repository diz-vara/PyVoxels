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
from pyquaternion import Quaternion
import struct


#%%
def read_csv(filename, delim = ' '):
    with open (filename) as csvfile:
        pclreader = csv.reader(csvfile, delimiter = delim)
        points = []
        for row in pclreader:
            pnt = np.array([float(p) for p in row if len(p) > 0])
            if (len(pnt) > 0):
                points.append(pnt)  

    return np.array(points)
    


#%%
def read_matrix(filename, delim = ' '):
    with open (filename) as csvfile:
        pclreader = csv.reader(csvfile, delimiter = delim)
        points = []
        for row in pclreader:
            pnt = np.array([float(p) for p in row if len(p) > 0])
            if (len(pnt) > 0):
                points.append(pnt)  


    matrix = np.matrix(points)     
    return matrix
    

#%%
def save_matrix(filename, mat):
        with open(filename, 'wt') as out_file:

            for row in range (mat.shape[0]):
                for col in range (mat.shape[1]):
                    out_file.writelines("{} ".format(mat[row,col]))
                out_file.write("\r\n")    
        out_file.close();
                
    

#%%
def read_cloud_csv(cloud_num, base_dir = 'E:\\Data\\Voxels\\London-cal1\\selected\\'):
    fname = base_dir + '\\{:06}.csv'.format(cloud_num)
    return read_cloud_file(fname)

def read_cloud_file(fname, delimiter = ','):

    with open (fname) as csvfile:
        pclreader = csv.reader(csvfile, delimiter = delimiter)
        points = []
        for row in pclreader:
            if ( 'Orientation:' in row or 'INSPVA' in row):
                continue;
            if (len(row) >= 3):    
                pnt = np.array([float(p) for p in row])
                points.append(pnt[:3])

        if ( False and 'Orientation:' in row):
            d = {}
            for row in pclreader:
                d[row[0].split()[0][0]]=float(row[0].split()[1])
            q = Quaternion(w=d['w'], x = d['x'], y = d['y'], z = d['z'])
            #print(q)    
        else:
            q=None
                

    cloud = np.array(points)     
    #if (len(cloud) > 1):
    #    cloud[:,1] = cloud[:,1] * -1
        
    return cloud,q
    
#%%

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

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img
#%%
def read_image(num, base_dir = 'E:\\Data\\Voxels\\London-cal1\\selected_raw\\'):
    imgname = base_dir + '{:06d}.jpg'.format(num)
    print(imgname)
    img = cv2.imread(imgname,-1)
    return img
    
    
 #%%
def find_image_corners(num,base_dir=None,ax = None):
    if (base_dir is None):
        base_dir='E:\\Data\\Voxels\\London-cal1\\selected_raw\\'
    img = read_image(num, base_dir)
    grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,corners=cv2.findChessboardCorners(grey,(11,11))
    
    if (not ret):
        corners = None
    else:
        corners = corners[:,0,:]
    if (ax):
        cv2.drawChessboardCorners(img,(11,11), corners, ret)
        ax.imshow(grey,cmap='gray')
    return corners    

   
#%%
def read_timed_points(points_file, points_n = -1):
    
    point_format = 'ffff'
    point_size = struct.calcsize(point_format)
    
    points_num = os.path.getsize(points_file)/point_size;

    if (points_n > 0 and points_n < points_num):
        points_num = points_n;

    step = points_num//1000;

    

    pts = []
    cnt = 0;        
    with open(points_file,'rb') as p_file:
        while(cnt < points_num):
            buf = p_file.read(point_size);
            if (len(buf) < point_size):
                break;
            x,y,z,time = struct.unpack(point_format,buf);
            pnt = np.array([x,y,z,time]);
            pts.append(pnt);
            cnt += 1;
            if (cnt%step == 0):
                print(int(cnt*100/points_num),"%")

    return np.array(pts);


#%%

def filter_points(cloud, limits):
    fo =  ((cloud[:,2] > limits[2,0]) & (cloud[:,1]<limits[1,1]) & (cloud[:,1] > limits[1,0]) & (cloud[:,0] > limits[0,0]) & (cloud[:,0] < limits[0,1]) & (cloud[:,2] < limits[2,1]) )    
    return cloud[fo,:]

    