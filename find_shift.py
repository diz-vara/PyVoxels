# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 12:46:57 2018

@author: avarfolomeev
"""

import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

def find_shift(pwrpak, o_imu):
    
    proc_len = 20000
    
    
    time_p = np.array([i['time'] for i in pwrpak[:proc_len]])
    time_o = np.array([i['time'] for i in o_imu[:proc_len]])
    time_pp = time_p - time_p[0]
    time_oo = time_o - time_o[0]

    accP = np.array([p['accY'] for p in pwrpak[:proc_len]])
    accP = accP*0.2/65536/125
    accO = np.array([o['accX'] for o in o_imu[:proc_len]])

    f_pwrpak = interpolate.interp1d(time_pp, accP, kind='cubic')
    f_ouster  =  interpolate.interp1d(time_oo, accO, kind='cubic')
    
    time_min = np.min((time_pp[-1], time_oo[-1]))
    
    time_new = np.arange(0, time_min, .004)
    y_p = f_pwrpak(time_new)
    y_o = f_ouster(time_new)

    cr = plt.xcorr(y_p,-y_o,maxlags=8000)
		
    shift = time_new[cr[0][np.argmax(cr[1])]]

    return shift,cr
