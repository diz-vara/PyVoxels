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

camera = 'cam0'
angle = np.radians(180-45); # 180-45, 180, 180+45 


base_dir = 'e:/data/Voxels/201902_USA/20190211_calibration/'
data_dir = base_dir + 'l_' + camera + '/s/'
cam_cal = pickle.load(open(base_dir + camera + '_subpix.p','rb'))

mtx = cam_cal[1]
dist = cam_cal[2]

csvlist = glob.glob(data_dir + '/*.csv')
jpglist = glob.glob(data_dir + '/*.jpg')

#names check

csvnames = [os.path.basename(csv).split('.')[0] for csv in csvlist]
jpgnames = [os.path.basename(jpg).split('.')[0] for jpg in jpglist]

assert (csvnames == jpgnames)




#%%
def _proc_cal_idx(idx):
    idx = np.array(idx)
  
    
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            _cloud,_grid = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.111, 9, 0.140),
                                             delimiter = ','); #' ' for cam2!
            _board = load_draw_2d_board(jpglist[cl],False,ax1,(9,9));

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
    scatt2d(ax1,imgpts[:,0,:],False,'w',size=15)
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
                                             _grid=(0.111, 9, 0.140),
                                             delimiter = ','); #' ' for cam2!
            _board = load_draw_2d_board(jpglist[cl],False,ax1,(9,9));

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
    scatt2d(ax1,imgpts[:,0,:],False,'w',size=15)
    print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
    return rot,t

#%%

rot,t = _proc_cal(2)

cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, 
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
        
