# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 12:12:57 2018

@author: avarfolomeev
"""

clouds = np.empty((0,3),np.float64)
boards = np.empty((0,2),np.float64)



ax1.cla()
ax3.cla()


base_dir = 'e:/data/Voxels/201809_usa/test15_6-camera_calibration/cam_1/png'

#cloud_list = [2,3,4,5,6,8,11,14,16]
cloud_list = [2,4,11]

for cl in cloud_list:
    _cloud,_grid = calc_cloud_grid(cl,base_dir,ax3, True);
    _board = calc_image_grid(cl,base_dir,False,ax1); #true for back only!!!
 
    print (len(_grid), len(_board))

    clouds=np.append(clouds, _grid ,0);
    boards=np.append(boards, _board,0)

ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist,flags=cv2.SOLVEPNP_ITERATIVE)
imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
scatt2d(ax1,imgpts[:,0,:],False,'w',size=15)
