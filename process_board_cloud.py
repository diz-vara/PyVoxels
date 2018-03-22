# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:14:49 2018

@author: avarfolomeev
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitali
import scipy.linalg
import cv2
import numpy.linalg as la

import read_csv

def fit_line(line_points, step = 0.1):
    num_points = line_points.shape[0]
    A = np.concatenate((line_points, np.ones((num_points,1))), axis=1)
    f = np.linspace(step, step*num_points, num_points);
    C, _, _, _ = scipy.linalg.lstsq(A, f)
    return C
    
 
def fit_line_svd(line_points):
     points = line_points.copy()
     avg = np.mean(points,0)
     _,_,V = la.svd(points)
     direction = V[:,1]
     t = np.linspace(-1,1,21)
     P = [avg + tt*direction for tt in t]
     
  
#%%

#calculates 
def fit_plane(data_):

    data = data_.copy()
    avg = np.mean(data,0)
    for i in range(data.shape[0]):
        data[i] = data[i] - avg

    A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients
    return C, avg
        
#%%

def plot_line(ax,p0,v,color='b',m=''):
    ax.plot([p0[0],p0[0]+v[0]], [p0[1], p0[1]+v[1]], [p0[2], p0[2]+v[2]],
               c=color,marker=m)
    
    
    
#%%
def plot_and_fit(ax, plane, color='b'):
    scatt3d(ax,plane,False,color)
    X,Y = np.meshgrid(np.arange(-.5, .5, 0.1), np.arange(-.5, .5, 0.1))
    C, avg = fit_plane(plane)


    Z = C[0]*X + C[1]*Y + C[2]
    ax.plot_surface(X+avg[0], Y+avg[1], Z+avg[2], rstride=1, cstride=1, alpha=0.2,
                   color=color)
    cx = C; #np.array([C[2],C[0],C[1]])
    c = cx /np.linalg.norm(cx)
    p0 = np.array([X[0,0],Y[0,0],Z[0,0]])
    p1 = np.array([X[-1,0],Y[-1,0],Z[-1,0]])
    p3 = np.array([X[0,-1],Y[0,-1],Z[0,-1]])
    p2 = np.array([X[-1,-1],Y[-1,-1],Z[-1,-1]])
    plot_line(ax,p0+avg,c,color,'o')
    plot_line(ax,p1+avg,c,color,'s')
    plot_line(ax,p2+avg,c,color,'*')
    plot_line(ax,p3+avg,c,color,'d')
    v0 = p2-p0
    v1 = p3-p1
    nrm = np.cross(v0,v1)
    nrm = nrm/np.linalg.norm(nrm)
    plot_line(ax,avg,c,color,'o')
    plot_line(ax,avg,nrm,color,'*')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax_3d.set_zlim(-1,1)
    ax.axis('equal')

    return c, avg, np.array([p0,p1,p2,p3])
#%%
import numpy.linalg as la

def find_rotation(a,b):
    a = a/la.norm(a);
    b = b/la.norm(b);

    v = np.cross(a,b)
    s = la.norm(v)
    c = np.dot(a,b)
    I = np.eye(3)
    
    vx = np.matrix([[0, -v[2], v[1]],[v[2],0,-v[0]], [-v[1], v[0],0]])
    r = I + vx + (vx * vx) * (1-c) / (s*s)
    return r
    
#%%
#calculates cloud-plane rotation to horizontal (x-y) plane,
#returns middle-point and rotation matrix
def get_cloud_rotation(cloud):
    X,Y = np.meshgrid(np.arange(-.5, .5, 0.1), np.arange(-.5, .5, 0.1))
    C, avg = fit_plane(cloud)
    Z = C[0]*X + C[1]*Y + C[2]
    p0 = np.array([X[0,0],Y[0,0],Z[0,0]])
    p1 = np.array([X[-1,0],Y[-1,0],Z[-1,0]])
    p2 = np.array([X[-1,-1],Y[-1,-1],Z[-1,-1]])
    p3 = np.array([X[0,-1],Y[0,-1],Z[0,-1]])
    v0 = p2-p0
    v1 = p3-p1
    nrm = np.cross(v0,v1)
    nrm = nrm/np.linalg.norm(nrm)

    target = np.array([0,0,1]); 
    rot = find_rotation(nrm, target);
    return avg, rot    

def rotate_cloud(cloud,avg,rot):
    result = (cloud-avg)*rot.transpose();
    
    filt = np.array(abs(result[:,2]) < 0.04).flatten();
    return np.array(result[filt,:])

#%%
def get_box(rotated_cloud):
        flat_cloud = rotated_cloud[:,:2].astype(np.float32)
        bounding_rect=cv2.minAreaRect(flat_cloud)
        box=cv2.boxPoints(bounding_rect)
        return box
        
#returns rotation point (bottom--left)        
def get_box_rotation(box):

    p0 = bottom_left(box)
    p1 = top_left(box)    
        
    a = np.append(p1-p0,0)
    b = np.array([0.,1.,0.])
    
    rot = find_rotation(b,a)
    
    return np.append(p0,0),rot


def bottom_left(box):
        p0 = [1e9, 1e9];
        for p in box:
            if p[0] < p0[0] and p[1] < p0[1]:
                p0 = p
        return p0
        
def top_left(box):
        p0 = [1e9, -1e9];
        for p in box:
            if p[0] < p0[0] and p[1] > p0[1]:
                p0 = p
        return p0
        
#%%
def calc_cloud_grid(num,ax=None):
    cloud = read_cloud_csv(num)
    new_axes = np.array([1,2,0]);
    cloud = cloud[:,new_axes]
    
    avg, rot = get_cloud_rotation(cloud)
    
    flat = rotate_cloud(cloud,avg,rot)
    
    box = get_box(flat)
    
    p0,box_rot = get_box_rotation(box)
    
    markers = np.arange(0.14,0.56,0.04)
    grid = np.array([(i,j,0)  for i in markers for j in markers])
    
    #first rotation - to flat box
    rotated_grid = grid*box_rot.transpose()+p0
    
    #second rotation - to the original box image
    rotated_grid = np.array(rotated_grid*la.pinv(rot).transpose() + avg)
    

    old_axes = np.array([2,0,1])
    
    o_cloud = cloud[:,old_axes]
    o_rotated_grid = rotated_grid[:,old_axes]    

    if (ax):
        scatt3d(ax,o_cloud,True,'b')
        scatt3d(ax,o_rotated_grid,False,'r')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #ax_3d.set_zlim(-1,1)
        #ax.axis('equal')

    return o_cloud, o_rotated_grid
     
   