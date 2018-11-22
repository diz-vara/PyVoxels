# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 12:06:52 2018

@author: avarfolomeev
"""
import numpy as np
from pwrpak_reader import *


def get_times_acc(pwrpak, imu):
    p_t_ = np.array([i['time'] for i in pwrpak])
    accY_ = np.array([i['accY'] for i in pwrpak])
    o_t = np.array([o['time'] for o in imu])
    o_x = np.array([o['accX'] for o in imu])

    return {"pwrTime":p_t_, 
            "ousterTime":o_t,
            "pwrAcc":accY_,
            "ousterAcc":o_x}

 #%%
 pwrpak155 = read_ggaex_log('e:/Data/Voxels/201809_usa/test15_5/IMU/test15_5.ggaex1')
 o_imu155 = ouster_imu_reader('e:/data/Voxels/201809_usa/test15_5/IMU/timestamps.txt')
 times155 = get_times_acc(pwrpak151, o_imu151)

 ax1.cla() 
 ax1.plot(times155['pwrTime'][:70000]-times155['pwrTime'][0],times155['pwrAcc'][:70000], marker='.', c='r')
 ax1.plot(times155['ousterTime'][:50000]-times155['ousterTime'][0],-(times155['ousterAcc'][:50000])*5+1, marker='.', c='b')
 
 
 

 op=times155['ousterTime'][0]-times155['pwrTime'][0]
 shift = 436.925 - 438.669   #ouster - pwrPack!!!!
 op_shift = op+shift
 
 
 ax1.cla()
 ax1.plot(times155['ousterTime'][:50000],-(times155['ousterAcc'][:50000])*5+1, marker='.', c='b')
 ax1.plot(times155['pwrTime'][:70000]+op_shift,times155['pwrAcc'][:70000], marker='.', c='r')
 #%%
 pwrpak115 = read_ggaex_log('e:/Data/Voxels/201809_usa/test11_5/IMU/test11_5.ggaex1')
 o_imu115 = ouster_imu_reader('e:/data/Voxels/201809_usa/test11_5/IMU/timestamps.txt')
 times115 = get_times_acc(pwrpak115, o_imu115)

 ax1.cla() 
 ax1.plot(times115['pwrTime'][:70000]-times115['pwrTime'][0],times115['pwrAcc'][:70000], marker='.', c='r')
 ax1.plot(times115['ousterTime'][:50000]-times115['ousterTime'][0],-(times115['ousterAcc'][:50000])*5+1, marker='.', c='b')
 
 
 

 op=times115['ousterTime'][0]-times115['pwrTime'][0]
 shift = 457.443 - 452.451   #ouster - pwrPack!!!!
 op_shift = op+shift
 
 
 ax1.cla()
 ax1.plot(times115['ousterTime'][:50000],-(times115['ousterAcc'][:50000])*5+1, marker='.', c='b')
 ax1.plot(times115['pwrTime'][:70000]+op_shift,times115['pwrAcc'][:70000], marker='.', c='r')
 