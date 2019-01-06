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

from read_csv import *

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
def sort_grid(rotated_grid, side_len = 11):
    #re-arrange rotated_grid 
    top_idx = np.argmax(rotated_grid[:,2])
    right_idx= np.argmax(rotated_grid[:,1])
    left_idx = np.argmin(rotated_grid[:,1])
    
    #print(top_idx, right_idx, left_idx)
    
    sign = np.sign(right_idx-top_idx)
    col_idx = np.arange(top_idx, right_idx+sign, int((right_idx-top_idx+sign)/(side_len-1)))

    sign = np.sign(left_idx-top_idx)
    row_idx = np.arange(0, (left_idx-top_idx+sign), int((left_idx-top_idx+sign)/(side_len-1)))
    
    idx = np.array([row + col for col in col_idx for row in row_idx])
    
    return (rotated_grid[idx])
    
#%%

def build_grid(sq_size, number, offset):
    markers = np.arange(number) * sq_size + offset
    
    grid = np.array([(0,i,j)  for i in markers for j in markers])
    return grid
    
    
#%%
def calc_cloud_grid(num, base_dir, ax=None, overlay=False,London=False, _grid = (0.0597, 11, 0.211)):
    fname = base_dir + '\\{:06}.csv'.format(num)
    return calc_cloud_grid_f(fname, ax, overlay, London, _grid)
    
def calc_cloud_grid_f(fname, ax=None, overlay=False,London=False, 
                      _grid = (0.0597, 11, 0.211),delimiter = ','):
    cloud = read_cloud_file(fname, delimiter)[0]
    #cloud = cloud * rot180
    _avg, _rot = get_cloud_rotation(cloud)
    
    flat = rotate_cloud(cloud,_avg,_rot)
    
    box = get_box(flat)
    
    p0,box_rot = get_box_rotation(box)
    
    if (London):
        grid =(0.04, 11, 0.14)# - LONDON

    grid = build_grid(_grid[0], _grid[1], _grid[2])
    
    #first rotation - to flat box
    rotated_grid = grid*box_rot.transpose()+p0
    
    #second rotation - to the original box image
    rotated_grid = np.array(rotated_grid * _rot + _avg)
    #rotated_grid = np.array(rotated_grid*la.pinv(rot).transpose() + avg)
    
    sorted_grid = sort_grid(rotated_grid,_grid[1])

    
    
    if (ax):
        scatt3d(ax,cloud,not overlay,'#1f1f1f','o',3)
        scatt3d(ax,[0,0,0],False)
        scatt3d(ax,sorted_grid,False,None,'d',49)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #left_border = np.ceil( max(cloud[:,1]) * 5.) / 5.
        #right_border = np.floor( min(cloud[:,1]) * 5.) / 5.
        #ax.set_ylim(right_border, left_border)
        #ax_3d.set_zlim(-1,1)
        #ax.axis('equal')
        for i in range(len(sorted_grid)):
            ax3.text(sorted_grid[i,0],sorted_grid[i,1],sorted_grid[i,2], str(i+1)  )


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
def draw_3d_board(cloud, ax=None, grid = (0.0597, 11, 0.211)):
    if (not ax is None):
        scatt3d(ax,cloud,False,'b','.',1)
        scatt3d(ax,[0,0,0],False,'b','o',70)


    _avg, _rot = get_cloud_rotation(cloud)
    
    
    flat = rotate_cloud(cloud,_avg,_rot)
    
    box = get_box(flat)
    
    p0,box_rot = get_box_rotation(box)
    grid = build_grid(grid[0], grid[1], grid[2])
    
    #first rotation - to flat box
    rotated_grid = grid*box_rot.transpose()+p0
    
    #second rotation - to the original box image
    rotated_grid = np.array(rotated_grid * _rot + _avg)
    #rotated_grid = np.array(rotated_grid*la.pinv(rot).transpose() + avg)
    
    sorted_grid = sort_grid(rotated_grid)

    if (not ax is None):
        scatt3d(ax,sorted_grid,False,None,'d',50)
    
        for i in range(len(sorted_grid)):
            ax3.text(sorted_grid[i,0],sorted_grid[i,1],sorted_grid[i,2], str(i+1)  )

    return sorted_grid
    
    
#%%


def calc_image_grid(num, base_dir,back,ax=None):
    fname = base_dir + '/{:06}.png'.format(num)
    print(fname);
    return load_draw_2d_board(fname,back,ax)
    

def load_draw_2d_board(name, back,ax=None, shape = None):
    img = cv2.imread(name,-1)
    return draw_2d_board(img, back, ax, shape)
    
def draw_2d_board(img, back=False, ax=None, shape = None):
    #if (not ax is None):
    #    ax.cla()

    #uimg=cv2.undistort(img,mtx,dist)

    if (not shape is None):
        r,corn_xy = cv2.findChessboardCorners(img,shape)
        cxy=np.array(corn_xy[:,0,:]).astype(np.float64)
    else:
        shape = (11,11) #LA recordings with manual corners
        corners = ( (img[:,:,0]<=0) & (img[:,:,1]<=0) & (img[:,:,2]> 254) )
        corn_xy=np.nonzero(corners.transpose())
        cxy=np.array(corn_xy).transpose().astype(np.float64)
    
    # for resized (enlarged) imnages!!!
    if (img.shape[0] > 1100):
        cxy = cxy / 2;
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    if (not ax is None):
        ax.imshow(img)
        
    cxy_u = cv2.undistortPoints(np.array([cxy]),mtx,dist, R=mtx)[0]
    

    #if (not ax is None):
    #    scatt2d(ax,cxy_u, False, None, 'o',50)
    
    top_idx = np.argmin(cxy_u[:,1])
    right_idx = np.argmax(cxy_u[:,0])
    
    top = cxy_u[top_idx]
    cxy_ut = cxy_u-top;
    
    right = cxy_ut[right_idx]
    angle = np.degrees(np.arcsin(right[1]/np.linalg.norm(right)))
    rm = np.matrix(cv2.getRotationMatrix2D((0,0), -angle,1)[:,:2])
    
    
    cxy_rot = np.array(cxy_ut * rm);
    
    order = np.argsort(cxy_rot[:,1])
    
    final_order = np.array([]);
    for i in np.arange(0,shape[0]*shape[1],shape[0]):
        row = cxy_rot[order[i:i+shape[1]]]
        row_order = np.argsort(row[:,0])
        final_order = np.append(final_order,order[row_order+i])
    
        
    if (back):    
        final_order = final_order.reshape((11,11)).transpose().reshape(-1)    
    cxy_ret = cxy[final_order.astype(np.int32)]
    
    if (not ax is None):
        scatt2d(ax, cxy_ret, False, None, 'd',30)
    
        for i in range(len(cxy_ret)):
            ax.annotate(str(i+1),np.array(cxy_ret[i]))
   
    return cxy_ret
#%% 
"""
rot180 = np.matrix(np.diag([-1,-1,1]))

clouds = np.empty((0,3),np.float64)
boards = np.empty((0,2),np.float64)

base_dir = 'e:/data/Voxels/201809_usa/test15_6-camera_calibration/cam2_again/'
#cloud_list = [1,2,3,4,5,7,8,9,10,11]
cloud_list = [1,4,5,8,10,11]


ax1.cla()
ax3.cla()


base_dir = 'e:/data/Voxels/201809_usa/test15_6-camera_calibration/cam_5/png'

cloud_list = [0,8,19]

for cl in cloud_list:
    _cloud,_grid = calc_cloud_grid(cl,base_dir,ax3, True);
    _board = calc_image_grid(cl,base_dir,True,ax1); #true for back only!!!
 
    print (len(_grid), len(_board))

    clouds=np.append(clouds, _grid ,0);
    boards=np.append(boards, _board,0)

ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist,flags=cv2.SOLVEPNP_ITERATIVE)
imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
scatt2d(ax1,imgpts[:,0,:],False,'w',size=15)

#%%    
    
imgpts, jac = cv2.projectPoints(g, rot, t, mtx, dist)

res_name = "rot_t_{:04d}.p".format(num)
pickle.dump({"rot":rot,"t":t},open(res_name,"wb"))

"""    