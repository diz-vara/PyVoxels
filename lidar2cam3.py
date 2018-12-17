# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 14:19:53 2018

@author: avarfolomeev
"""

rot180 = np.matrix(np.diag([-1,-1,1]))

clouds = np.empty((0,3),np.float64)
boards = np.empty((0,2),np.float64)

#base_dir = 'e:/data/Voxels/201809_usa/test15_6-camera_calibration/cam2_again/'
#cloud_list = [1,2,3,4,5,7,8,9,10,11]
#cloud_list = [1,4,5,8,10,11]


ax1.cla()
ax3.cla()


base_dir = 'e:/data/Voxels/201809_usa/test15_6-camera_calibration/cam_3/'
cloud_list = [1,3,4,5,7,11,16,18]
cloud_list = [11, 16         ]

for cl in cloud_list:
    _cloud,_grid = calc_cloud_grid(cl,base_dir,ax3, True);
    _board = calc_image_grid(cl,base_dir,True,ax1); #true for back only!!!
    
    print (len(_grid), len(_board))
    
    clouds=np.append(clouds, _grid ,0);
    boards=np.append(boards, _board,0)


ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist,flags=cv2.SOLVEPNP_ITERATIVE)
imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
scatt2d(ax1,imgpts[:,0,:],False,'w',size=15)
