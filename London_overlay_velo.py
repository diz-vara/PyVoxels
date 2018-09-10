# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:36:31 2017

@author: avarfolomeev
"""

import scipy.misc
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np

    
#returns results in npoins by 4 byte array, 4th byte is for type
# first 3 bytes - color of the point

_diz2cs = np.array([
                   0,   #unmarked 
                   23,  #sky
                   7,   #road
                   8,   #sidewalk
                   65,  #lane marker
                   10,   #railway
                   22,   #terrain
                   21,   #vegetaion
                   11,   #building
                   15,   #--birdge
                   14,   #construction (guard rail)
                   20,   #sign
                   19,  #traffick light
                   26,  #transport (car)
                   24,  #pedestrian
                   25,  #rider
                   33,  #bicycle
                    5,  #animal (dynamic)
                    4   #static
                   ]).astype(np.uint8)

def London_overlay_velo(pnt, calibration, marks = False):
    
    v = dataset.velo(pnt)[::-1]
    ts = dataset.timestamps_v[pnt]
    
    image_idx = np.nonzero(ts_i>ts)[0][0]+2

    img = dataset.rgb(image_idx)
    img_u8 = (img*255).astype(np.uint8)
    img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGB2RGBA)


    if (dataset.get_road_num() > image_idx ):
        road_cs = _diz2cs[dataset.road(image_idx)]    
        img_u8[:,:,3] =  road_cs;
        
  
    

    npoints = len(v)


    vi = v.copy()
    vi[:,1] = 0. - vi[:,1]


    cloud,colors, idx = pykitti.get_colored_cloud(vi[:,:3], img_u8,
                                                  calibration)
    
    
    colored = np.zeros((npoints,4),dtype=np.uint8)
    colored [idx,:] = colors

    return v,colored
    
    
    