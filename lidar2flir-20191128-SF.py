# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 12:12:57 2018

@author: avarfolomeev
"""

import matplotlib.pyplot as plt
import pickle
import glob
import os
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitali
import numpy as np


from calibrate_camera import *
from process_board_cloud import *
from calib2json import *

from scatt3d import *


#%%
#%%
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1.,
                  scale = 1, tvec = None, rvec = None, points_idx = None):
    
    #pts_idx = np.arange(121);                       
    pts_idx=[0,10,110,60,120]
    #pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1


    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            name = jpglist[cl];
            img = cv2.imread(name,-1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
                                               
            #imgU = img.copy()
            #mtxU = mtx.copy()
            
            #imgU = cv2.undistort(img, mtx, dist, newCameraMatrix=mtxU)
                                                        
                                                        
            _board = load_draw_2d_board(name,mtx, dist, 
                                        back,ax1,(11,11), 
                                        scale = 1.);

            #_cloud = _cloud * rm90

            #_grid = _grid[pts_idx]
            #_board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    if (len(clouds) > 0):
        
        if (scale is not None and scale > 0):
            boards = boards * scale
        else:
            scale = 1.
    
        if (points_idx is not None):
            clouds = clouds[points_idx,:]
            boards = boards[points_idx,:]
            
            

        ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist, 
                                   rvec, tvec, 
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        
        
        imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
    
        bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
        
        scatt2d(ax1,imgpts[:,0,:]/scale,False,'w','x',size=25)
        scatt2d(ax1,bpts[:,0,:]/scale,False,'b','.',size=1)
        
        sum = cv2.norm(imgpts[:,0,:], boards);
        print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
        print ('reproj error =', sum/len(boards))
    else:
        print ("no good boards")
        rot = t = None
    return rot,t

    
#%%

f1 = plt.figure(1)
f3 = plt.figure(3)

ax1= f1.gca()
ax3=f3.add_subplot(111,projection='3d')


base_dir = 'e:/Data/Voxels/201911_sf/cal_20191128/'

#from process_board_cloud import *

#cameras  1   2    3    4    5
#angles : 0, 55, 152, 208, 305
cameras = ['cam_1', 'cam_2', 'cam_3', 'cam_4', 'cam_5']

#angles_degrees = [0, 55, 152, 208, 305]
#'cause we'are upside-down
angles_degrees = [0, 305, 208, 152, 55]

cam = 2

scale = 0.5
suffix = "_small"


camera = cameras[cam]
angle = np.radians(angles_degrees[cam])
#base_dir = 'e:/data/Voxels/201903_USA/20190321_cal_cam1/'
cam_cal = pickle.load(open(base_dir + camera + suffix + '.p','rb'))



copy_pictures(data_dir)



mtx = cam_cal[1]
dist = cam_cal[2]

data_dir = base_dir + 'lidar_' + camera + '/'
csvlist = glob.glob(data_dir + 's/*.txt')
jpglist = glob.glob(data_dir + 's/*.jpg')

#names check

csvnames = [os.path.basename(csv).split('.')[0] for csv in csvlist]
jpgnames = [os.path.basename(jpg).split('.')[0] for jpg in jpglist]

assert (csvnames == jpgnames)


lidar_rotation = eulerAnglesToRotationMatrix(np.radians([180,0,angles_degrees[cam]]))

rot,t = _proc_cal_idx([4,6,10,1], rotate = lidar_rotation, VLP32_multi=1.,
                      scale = scale)



cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, 
                   "camera_name":camera}    
                   
pickle.dump(cam_calib_dict,open(base_dir+"Flir_"+camera+"_dict_81.p","wb"))

        
save_calib_json(base_dir+"Flir_"+camera+"_t.json", cam_calib_dict)

           
cam_4_dict_81 = cam_calib_dict

#%%

if False:
    for jpg in jpglist:
        _board = load_draw_2d_board(jpg,False,ax1,(9,9));
        if ( _board is None):
            print ("--- ", jpg)
        else:
            print (jpg, len(_board))
        
