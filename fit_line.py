# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:14:49 2018

@author: avarfolomeev
"""

import scipy.linalg
import numpy as np


def fit_line(line_points, step = 0.1):
    num_points = line_points.shape[0]
    A = np.concatenate((line_points, np.ones((num_points,1))), axis=1)
    f = np.linspace(step, step*num_points, num_points);
    C, _, _, _ = scipy.linalg.lstsq(A, f)
    return C
    
 
def fit_line_svd(line_points):
     points = line_points.copy()
     avg = np.mean(points,0)
     _,_,V = np.linalg.svd(points)
     direction = V[:,1]
     t = np.linspace(-1,1,21)
     P = [avg + tt*direction for tt in t]
     
  
#%%

import scipy.linalg
import numpy as np

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
Y,Z = np.meshgrid(np.arange(-.5, .5, 0.1), np.arange(-.5, .5, 0.1))

C, avg = fit_plane(plane)

X = C[0]*Y + C[1]*Z + C[2]

ax_3d.cla()
ax_3d.plot_surface(X+avg[0], Y+avg[2], Z+avg[1], rstride=1, cstride=1, alpha=0.2)
#ax_3d.scatter(plane[:,0], plane[:,2], plane[:,1], c='b', s=50)
ax_3d.scatter(lineLL[:,0], lineLL[:,1], lineLL[:,2], c='r', s=50)
ax_3d.scatter(lineUL[:,0], lineUL[:,1], lineUL[:,2], c='c', s=50)
ax_3d.scatter(lineLR[:,0], lineLR[:,1], lineLR[:,2], c='y', s=50)
ax_3d.scatter(lineUR[:,0], lineUR[:,1], lineUR[:,2], c='g', s=50)
ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Y')
ax_3d.set_zlabel('Z')
ax_3d.axis('equal')

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
    p2 = np.array([X[0,-1],Y[0,-1],Z[0,-1]])
    p3 = np.array([X[-1,-1],Y[-1,-1],Z[-1,-1]])
    plot_line(ax,p0+avg,c,color,'o')
    plot_line(ax,p1+avg,c,color,'s')
    plot_line(ax,p2+avg,c,color,'*')
    plot_line(ax,p3+avg,c,color,'d')
    v0 = p1-p0
    v1 = p3-p0
    nrm = np.cross(v0,v1)
    nrm = nrm/np.linalg.norm(nrm)
    plot_line(ax,avg,c,color,'o')
    plot_line(ax,avg,nrm,color,'*')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax_3d.set_zlim(-1,1)
    ax.axis('equal')

    return c, avg, nrm
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
    

    