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
