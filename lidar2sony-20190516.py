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
def _proc_cal_idx(idx, mtx, dist):
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
            _board = load_draw_2d_board(jpglist[cl],False,ax1,(9,9),
                                        mtx,dist);

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
def _proc_cal_idx(idx, rotate = None):
    
    #pts_idx = np.arange(81);                       
    pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    #idx = cam0_idx
    
    
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            _cloud,_grid = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.111, 9, 0.140),
                                             delimiter = ',',rotate=rotate); #' ' for cam2!
            _board = load_draw_2d_board(jpglist[cl],mtx, dist, False,ax1,(9,9));

            #_cloud = _cloud * rm90

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
    
    sum = cv2.norm(imgpts[:,0,:], boards);
    print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
    print ('reproj error =', sum/len(boards))
    return rot,t

    
#%%

f1 = plt.figure(1)
f3 = plt.figure(3)

ax1= f1.gca()
ax3=f3.add_subplot(111,projection='3d')


base_dir = 'e:/data/Voxels/20190410_cal/'
#base_dir = 'e:/data/Voxels/20190516_cal_cam1/'
r180=np.matrix([[-1,0,0],[0,-1,0],[0,0,1]])
#from process_board_cloud import *

camera = 'cam_0'
angle = np.radians(0); #180+90); # 180-45, 180, 180+45 


#base_dir = 'e:/data/Voxels/201903_USA/20190321_cal_cam1/'
data_dir = base_dir + 'l-' + camera + '/s/'
cam_cal = pickle.load(open(base_dir + camera + '_subpix.p','rb'))

mtx = cam_cal[1]
dist = cam_cal[2]

csvlist = glob.glob(data_dir + '/*.csv')
jpglist = glob.glob(data_dir + '/*.jpg')

#names check

csvnames = [os.path.basename(csv).split('.')[0] for csv in csvlist]
jpgnames = [os.path.basename(jpg).split('.')[0] for jpg in jpglist]

assert (csvnames == jpgnames)



rot,t = _proc_cal_idx([0,1])
#rot,t = _proc_cal_idx([0,1,5,6,7,8,9,10,20,19,17,15,13],r180)
#[[ 1.33418012  1.4054811  -1.13807515]] 
# [[-0.04419182 -0.10124581 -0.2294884 ]]
#reproj error = 0.20974668691942583

cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, 
                   "camera_name":"argus_"+ camera}    
                   
pickle.dump(cam_calib_dict,open(base_dir+"uk_"+camera+"_dict_9.p","wb"))

        
save_calib_json(base_dir+"uk_"+camera+"_9.json", cam_calib_dict)

           
cam_1_dict_81 = cam_calib_dict

#%%

if False:
    for jpg in jpglist:
        _board = load_draw_2d_board(jpg,False,ax1,(9,9));
        if ( _board is None):
            print ("--- ", jpg)
        else:
            print (jpg, len(_board))
        
