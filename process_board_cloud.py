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

    #fit X against Y-Z  plane!!!
    A = np.c_[data[:,1], data[:,2], np.ones(data.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,0])    # coefficients
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
    Y,Z = np.meshgrid(np.arange(-1, 1.01, 0.1), np.arange(-1, 1.01, 0.1))
    C, avg = fit_plane(cloud)
    X = C[0]*Y + C[1]*Z + C[2]

    p0 = np.array([X[0,0],Y[0,0],Z[0,0]])
    p1 = np.array([X[-1,0],Y[-1,0],Z[-1,0]])
    p2 = np.array([X[-1,-1],Y[-1,-1],Z[-1,-1]])
    p3 = np.array([X[0,-1],Y[0,-1],Z[0,-1]])
    v0 = p2-p0
    v1 = p3-p1
    nrm = np.cross(v0,v1)
    nrm = nrm/np.linalg.norm(nrm)
    #print(v0,v1,nrm)
    
    target = np.array([1,0,0]); 
    rot = find_rotation(nrm, target);
    return avg, rot    

def rotate_cloud(cloud,avg,rot):
    result = (cloud-avg)*rot.transpose();
    
    filt = np.array(abs(result[:,0]) < 0.1).flatten();
    return np.array(result[filt,:])

#%%
def get_box(rotated_cloud):
        flat_cloud = rotated_cloud[:,1:].astype(np.float32)
        bounding_rect=cv2.minAreaRect(flat_cloud)
        box=cv2.boxPoints(bounding_rect)
        return box
        
#returns rotation point (bottom--left)        
def get_box_rotation(box):

    p0 = bottom_left(box)
    p1 = top_left(box)    
        
    a = np.append(0,p1-p0)
    b = np.array([0.,0.,1.])
    
    rot = find_rotation(b,a)
    
    return np.append(0,p0),rot


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
# re-sorts grid in order top corner->right corner
def sort_grid(rotated_grid):
    #re-arrange rotated_grid 
    top_idx = np.argmax(rotated_grid[:,2])
    right_idx= np.argmax(rotated_grid[:,1])
    left_idx = np.argmin(rotated_grid[:,1])
    
    #print(top_idx, right_idx, left_idx)
    
    sign = np.sign(right_idx-top_idx)
    col_idx = np.arange(top_idx, right_idx+sign, int((right_idx-top_idx+sign)/10))

    sign = np.sign(left_idx-top_idx)
    row_idx = np.arange(0, (left_idx-top_idx+sign), int((left_idx-top_idx+sign)/10))
    
    idx = np.array([row + col for row in row_idx for col in col_idx])
    
    return (rotated_grid[idx])
    
#%%

def build_grid(sq_size, number, offset):
    markers = np.arange(number) * sq_size + offset
    
    grid = np.array([(0,i,j)  for i in markers for j in markers])
    return grid
    
    
#%%
def calc_cloud_grid(num,ax=None):
    cloud = read_cloud_csv(num,base_dir)[0]
    
    _avg, _rot = get_cloud_rotation(cloud)
    
    flat = rotate_cloud(cloud,_avg,_rot)
    
    box = get_box(flat)
    
    p0,box_rot = get_box_rotation(box)
    
    grid = build_grid(0.0597, 11, 0.211)
    
    #first rotation - to flat box
    rotated_grid = grid*box_rot.transpose()+p0
    
    #second rotation - to the original box image
    rotated_grid = np.array(rotated_grid * _rot + _avg)
    #rotated_grid = np.array(rotated_grid*la.pinv(rot).transpose() + avg)
    
    sorted_grid = sort_grid(rotated_grid)

    
    
    if (ax):
        scatt3d(ax,cloud,True,'#1f1f1f','o',3)
        scatt3d(ax,[0,0,0],False)
        scatt3d(ax,sorted_grid,False,None,'d',49)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        left_border = np.ceil( max(cloud[:,1]) * 5.) / 5.
        right_border = np.floor( min(cloud[:,1]) * 5.) / 5.
        ax.set_ylim(left_border, right_border)
        #ax_3d.set_zlim(-1,1)
        #ax.axis('equal')


    #rotated_cloud = cloud * box_rot.transpose
        
    return cloud, sorted_grid
     
#%%
if 0:
    num=570
    c,g = calc_cloud_grid(num)
    
    
    corners = find_image_corners(num)
    
    
    ret, rot, t = cv2.solvePnP(g,corners,mtx,dist,flags=cv2.SOLVEPNP_ITERATIVE)
    imgpts, jac = cv2.projectPoints(g, rot, t, mtx, dist)
    
    res_name = "rot_t_{:04d}.p".format(num)
    pickle.dump({"rot":rot,"t":t},open(res_name,"wb"))


#%%

   