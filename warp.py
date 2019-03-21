# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 22:54:00 2017

@author: Anton Varfolomeev
"""

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
#dist_pickle = pickle.load( open( "mtx_dist.p", "rb" ) )
#mtx = dist_pickle["mtx"]
#dist = dist_pickle["dist"]


src = np.float32(
   [[ 337, 518],
    [ 334, 831], 
    [ 770, 515],
    [ 805, 824]])

left = 10
top = 25
dst = np.float32(
   [[ left-4.27-2.7, top-17.08],
    [ left-1.34, top-5.08], 
    [ left+2.75, top-17.58], 
    [ left+1.05, top-5.2]])

M = cv2.getPerspectiveTransform(src, dst*50)
Minv = cv2.getPerspectiveTransform(dst, src)

            # e) use cv2.warpPerspective() to warp your image to a top-down view
#image0 = cv2.imread('E:\\Data\\Voxels\\201902_USA\\20190216_151511_\\argus_cam_1\\data\\1550358975.579326630.jpg')
#image = cv2.undistort(image0, mtx, dist, None, mtx)
#warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
#plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))





#%%

def unpersp(base_dir, in_dir, out_dir, _flag):
    """Load road masks from file.
    Images are RGB, with:
        (255,0,255) for road
        (255,0,0) for not_road
    """
    
    if ( _flag is None or _flag == 0):
        flags = cv2.INTER_LINEAR
    else:
        _flag = 1
        flags = cv2.INTER_NEAREST

    in_dir = os.path.join(base_dir, in_dir)
    out_dir = os.path.join(base_dir, out_dir)

    try:
        os.makedirs(out_dir)
    except:
        pass


    print('Loading images from ' + in_dir + " writing to " + out_dir)

    
    

    im_files = sorted(os.listdir(in_dir))
    cnt = 0
    end_string = ' from ' + str(len(im_files))
    
    
    
    for _file in im_files:
        print(str(cnt) + end_string)
        cnt = cnt+1
        img_in = cv2.imread(os.path.join(in_dir, _file),-1)

        
        #undistort!!!
        und = cv2.undistort(img_in, cam_1_dict['mtx'], cam_1_dict['dist'])
        warped = cv2.warpPerspective(und, M, (img_in.shape[1], img_in.shape[0]), 
                                                 flags=flags)
        


        cv2.imwrite(os.path.join(out_dir, _file),warped)
    




