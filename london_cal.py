# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:50:19 2017

@author: avarfolomeev
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pykitti
import cv2
import labels

#%%
basedir='E:/Data/Voxels/'
date = 'London-cal1'
drive = '0015'
Marks = False
base=0
frame_range = range(300,600) #None; #range(base,base+2000)#total 4544
dataset = pykitti.raw(basedir, date, drive, frame_range)
#dataset.load_gray()
dataset.load_velo()
dataset.load_rgb()
dataset.load_calib()

Prect = dataset.calib.P_rect_20;
Rrect = dataset.calib.R_rect_20;
T_cam_velo = dataset.calib.T_cam2_velo;

runfile('plotVelo.py')
runfile('setFigure.py')

ax1 = f2.add_subplot(121,projection='3d')

#%%

veloTdir = os.path.join(basedir,date,dataset.drive,'velorgbTcsm')

try:
    os.makedirs(veloTdir)
except:
    pass

npoints = len(dataset.velo)

for i in range(npoints):
    print (dataset.drive,i,'from',npoints)
    filename = "{:010d}.velorgbTcs".format(i+base)
    filepath = os.path.join(veloTdir,filename)
    v,c = overlay_velo(i, Marks)

    save_velo_color(v,c,filepath)