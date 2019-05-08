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


nSamples= 18

f1 = plt.figure(1)
f3 = plt.figure(3)

ax1= f1.gca()
ax3=f3.add_subplot(111,projection='3d')


#from process_board_cloud import *

camera = 'cam_0'
cam_cal = pickle.load(open('e:/Data/Voxels/camera_cal/002/sony_' + camera + '.p','rb'))

mtx = cam_cal[2][1]
dist = cam_cal[2][2]
base_dir = 'e:/data/Voxels/201901-LA/20190119_calibration/' + camera + '/selected'
data_dir = 'e:/data/Voxels/201901-LA/20190119_calibration/' + camera + '/data1'

base_dir = 'E:\\Data\\Voxels\\201902_USA\\20190219_calibration1\\argus_cam_0\\s' 
csvlist = glob.glob(base_dir + '/*.csv')
jpglist = glob.glob(base_dir + '/*.jpg')

#names check

csvnames = [os.path.basename(csv).split('.')[0] for csv in csvlist]
jpgnames = [os.path.basename(jpg).split('.')[0] for jpg in jpglist]

assert (csvnames == jpgnames)

cam0_idx = [0,1,2,3,4,5,6,7,8,10,11,12,13,14,15]

#full list!@!
cam1_idx = [0,1,2,3,4,5,6,7,8,9,10]

#full list!@!
cam2_idx = [0, 1, 2, 3, 4, 5,6, 7, 8, 9, 10, 11, 12, 13, 14,15,16, 17, 18, 19, 20, 21]



#%%

def _proc_cal_idx(idx):
    
    pts_idx = np.array([1,6,11,56,61,66,111,116,121])-1
    #idx = cam0_idx
    
    
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            _cloud,_grid = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ','); #' ' for cam2!
            _board = load_draw_2d_board(jpglist[cl],False,ax1,(11,11));

            _grid = _grid[pts_idx]
            _board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    
    ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist,flags=cv2.SOLVEPNP_ITERATIVE)
    imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
    scatt2d(ax1,imgpts[:,0,:],False,'w','x',size=25)
    print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
    return rot,t

def _proc_cal(_nSamples=-1):
    idx = np.arange(len(csvlist)).astype(int)
    
    
    #idx = cam0_idx
    
    np.random.shuffle(idx)
    if (_nSamples > 0):
        idx = idx[:_nSamples]
    
    
    
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            _cloud,_grid = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ','); #' ' for cam2!
            _board = load_draw_2d_board(jpglist[cl],False,ax1,(11,11));

            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    
    ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist,flags=cv2.SOLVEPNP_ITERATIVE)
    imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
    scatt2d(ax1,imgpts[:,0,:],False,'w','x',size=25)
    print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
    return rot,t

#%%

rot,t = _proc_cal(20)

cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":np.radians(180), 
                   "camera_name":"argus_"+ camera}    
                   
pickle.dump(cam_calib_dict,open("sony_"+camera+"_dict.p",'wb'))

        
save_calib_json('sony_'+camera+".json", cam_calib_dict)

           
cam_2_dict = cam_calib_dict

#%%

if False:
    for jpg in jpglist:
        _board = load_draw_2d_board(jpg,False,ax1,(11,11));
        if ( _board is None):
            print ("--- ", jpg)
        else:
            print (jpg, len(_board))
        
