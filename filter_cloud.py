# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 09:19:33 2018

@author: avarfolomeev
"""

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
        

def filter_pts(pts):
    pts=pts.round()
    pts_idx= (pts[:,0] >= 0) & (pts[:,0]< 1920) & (pts[:,1] >= 0) & (pts[:,1] < 1200)
    return pts_idx
    
    
def get_colored_cloud(cloud, image):    
    f_cloud,idx = filter_cloud(cloud)
    pts, jac = cv2.projectPoints(f_cloud, rot, t, mtx, dist)
    pts = pts[:,0,:]
    pts_idx = filter_pts(pts)
    ff_cloud = f_cloud[pts_idx,:]
    f_pts = np.round(pts[pts_idx]).astype(int)
    colors = image[f_pts[:,1],f_pts[:,0]]
    idx = idx[pts_idx]
    return ff_cloud, colors, idx
    



