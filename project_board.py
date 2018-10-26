# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 17:13:03 2018

@author: avarfolomeev
"""
cl9 = read_cloud_csv(9,'e:/data/Voxels/201809_usa/test15_6-camera_calibration/cam_2_new_dist/')[0]
Img = cv2.imread('e:/data/Voxels/201809_usa/test15_6-camera_calibration/cam_2_new_dist/000000L.png ',-1)
Img = cv2.cvtColor(Img,cv2.COLOR_RGBA2BGR)

cl9f = cl9[cl9[:,0]>0]
pts, jac = cv2.projectPoints(cl9f, rot,t,mtx,dist)


pts = pts[:,0,:]



pts_idx = filter_pixels(pts, (1280,720))

f_cloud = cl9f[pts_idx,:]

f_pts = np.round(pts[pts_idx]).astype(int)

colors = Img[f_pts[:,1],f_pts[:,0]]

scatt3d(ax3,f_cloud,True,colors/255)
