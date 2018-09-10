# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 09:43:23 2018

@author: avarfolomeev
"""



i2l=[]


for i in range(24):
    i2l.append(get_imu_to_lidar_rotation(cl_q[i].rotation_matrix, get_road_rotation(clouds[i])[1]))