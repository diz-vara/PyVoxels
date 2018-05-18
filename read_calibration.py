# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:14:54 2018

@author: avarfolomeev
"""

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
               