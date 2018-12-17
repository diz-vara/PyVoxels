# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 14:16:04 2018

@author: avarfolomeev
"""
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitali


clouds = np.empty((0,3),np.float64)
boards = np.empty((0,2),np.float64)



ax1.cla()
ax3.cla()


base_dir = 'e:/data/Voxels/201809_usa/test15_6-camera_calibration/cam_5/png'

cloud_list = [200,8,3]

for cl in cloud_list:
    _cloud,_grid = calc_cloud_grid(cl,base_dir,ax3, True);
    _board = calc_image_grid(cl,base_dir,False,ax1); #true for back only!!!
 
    print (len(_grid), len(_board))

    clouds=np.append(clouds, _grid ,0);
    boards=np.append(boards, _board,0)

ret, rot, t = cv2.solvePnP(clouds,boards,cam5_mtx,cam5_dist,flags=cv2.SOLVEPNP_ITERATIVE)
imgpts, jac = cv2.projectPoints(clouds, rot, t, cam5_mtx, cam5_dist)
scatt2d(ax1,imgpts[:,0,:],False,'w',size=15)
