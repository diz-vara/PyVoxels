# -*- coding: utf-8 -*-
"""
Created on Fri May 18 12:53:47 2018

@author: avarfolomeev
"""

IMUdata21 = read_IMUfile('E:\\Data\\Voxels\\2018_03_08\\L21\\IMU\\data\\Data.imu')

points21_2, frames21=read_velo_file('E:\\Data\\Voxels\\2018_03_08\\L21\\velodyne_packets\\')

pt21_2,n21_2,f21_2, d_pos, q_int=translate_points(points21_2[:700000],velo_rot, IMUdata21)
pt21_2l = pt21_2 - d_pos;
pt21_2lr = np.dot(pt21_2l,q_int.rotation_matrix)
pt21_2lrv = np.dot(pt21_2lr,velo_rot)
cloud,colors,idx = l_get_colored_cloud(pt21_2lrv, img_u8,calibration)
full_colors = np.zeros((len(f21_2),4),np.uint8)

full_colors[idx]=colors
save_points('E:\\Data\\Voxels\\2018_03_08\\L21\\ff',pt21_2,f21_2,full_colors)
