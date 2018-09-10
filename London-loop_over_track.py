# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:50:19 2017

@author: avarfolomeev
"""


import os

os.chdir('d:/WORK/KITTI/pykitti')

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pykitti
import cv2
import labels
import pickle



#%%
calibration_dir = 'e:/Data/Voxels/London-cal1/results/'
mtx_dist_file = 'mtx_dist-tilt2-oCVexample_11x11_tilt.p'
rot_t_file = 'rot_t_collected.p'

mtx_dist = pickle.load(open(os.path.join(calibration_dir, mtx_dist_file),'rb'))

#mtx = mtx_dist['mtx']
#dist = mtx_dist['dist']

rot_t = pickle.load(open(os.path.join(calibration_dir, rot_t_file),'rb'))
#rot=rot_t['rot']
#t=rot_t['t']

calibration = {'mtx':mtx_dist['mtx'],
               'dist':mtx_dist['dist'],
                'rot':rot_t['rot'],
                't':rot_t['t']
               }
#%%

basedir='E:/Data/Voxels'
date = '2018_03_08'
drive = '0022'
Marks = False
frame_range = None; #range(base,4544) #None; #range(base,base+2000)#total 4544
dataset = pykitti.raw(basedir, date, drive, frame_range)
#dataset.load_gray()
dataset.load_velo()
dataset.load_rgb()
dataset.load_velo_timestamps()
dataset.load_image_timestamps()

ts_i = np.array(dataset.timestamps_i)

#dataset.load_calib()
dataset.load_Croad()

#Prect = dataset.calib.P_rect_20;
#Rrect = dataset.calib.R_rect_20;
#T_cam_velo = dataset.calib.T_cam2_velo;

#%%

veloTdir = os.path.join(basedir,date,dataset.drive,'velorgbTcsm')

try:
    os.makedirs(veloTdir)
except:
    pass

npoints = dataset.get_velo_num()

runfile('London_overlay_velo.py')


for i in range(npoints):
    print (dataset.drive,i,'from',npoints)
    filename = "{:010d}.velorgbTcs".format(i)
    filepath = os.path.join(veloTdir,filename)
    v,c = London_overlay_velo(i, calibration, Marks)

    pykitti.save_velo_color(v,c,filepath)
