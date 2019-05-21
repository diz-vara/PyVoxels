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
import sys
import glob
import tensorflow as tf

from read_frozen import read_frozen

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
Minv = cv2.getPerspectiveTransform(dst*50, src)

            # e) use cv2.warpPerspective() to warp your image to a top-down view
#image0 = cv2.imread('E:\\Data\\Voxels\\201902_USA\\20190216_151511_\\argus_cam_1\\data\\1550358975.579326630.jpg')
#image = cv2.undistort(image0, mtx, dist, None, mtx)
#warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
#plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))


cam_1_dict = pickle.load(open('/media/undead/Data/Voxels/sony_cam_1_dict.p','rb'))
cascadeArrows = cv2.CascadeClassifier('/media/undead/Data/Voxels/Arrows/cArr_24_1_50_s11.xml')

arrowsNet = read_frozen('/media/undead/Data/Voxels/Arrows/arrows-4.frozen_model.pb')

inp = arrowsNet.get_tensor_by_name('input_image:0')
keep_prob = arrowsNet.get_tensor_by_name('keep_prob:0')
out = arrowsNet.get_tensor_by_name('lin2:0')

arrow_types = ["BG", "LU", "LUR", "RU", "U", "dL", "dR", "diagL", "diagR", "uL", "uR"]

#%%
ROAD_MARKING = 7

def getArrows(base_dir, out_dir = "", start = 0, end = -1):
    if (base_dir is None):
        print('You must select base_dir')
        return
    
    in_dir = os.path.join(base_dir, 'data')
    mask_dir = os.path.join(base_dir, 'Xroad')



    print('Loading images from ' + in_dir)
    
    if (out_dir is None):
        out_dir = ""
    if (len (out_dir) > 0):
        try:
            os.makedirs(out_dir)
        except:
            pass

        for sub_dir in arrow_types:
            try:
                os.makedirs(os.path.join(out_dir,sub_dir))
            except:
                pass;
        
        print(" writing arrows to " + out_dir)
    
    im_files = sorted(glob.glob(in_dir+'/*.*g'))
    
    if (start is None):
        start = 0;
        
    if (end is None or end < 0):
        end = len(im_files)
    im_files = im_files[start:end+1]
    cnt = 0
    end_string = ' from ' + str(len(im_files))
    config = tf.ConfigProto()
    #config.gpu_options.visible_device_list = '1'
    warp_shape_0 = (0,0)

    with tf.Session(graph=arrowsNet, config=config) as sess:
    
        for _file in im_files:
            print(str(cnt) + end_string)
            cnt = cnt+1

            img_in = cv2.imread(os.path.join(in_dir, _file),-1) #GRAY only!!!
            grey = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    
            warp_shape = (img_in.shape[1], img_in.shape[0])
            
            if (warp_shape_0 != warp_shape):
                umap = cv2.initUndistortRectifyMap(cam_1_dict['mtx'],cam_1_dict['dist'],
                                                   None,cam_1_dict['mtx'],
                                                   warp_shape, cv2.CV_32FC1)
                warp_shape_0 = warp_shape
                

            o_file = _file.replace('.jpg', '_.jpg')

            _file = os.path.split(_file)[-1]
      
            #if (True):
            try:
                

                und = cv2.undistort(grey, cam_1_dict['mtx'], cam_1_dict['dist'])
                warped = cv2.warpPerspective(und, M, warp_shape, 
                                                         flags=cv2.INTER_LINEAR)
                
                
                detected = cascadeArrows.detectMultiScale(warped,
                                                          1.15,
                                                          minNeighbors=1,
                                                          minSize=(50,50),
                                                          maxSize=(160,160))
                
                if (len(detected) > 0):
                    print ( "Detected ", len(detected), " arrows")
                    mask = np.zeros(img_in.shape[:2], dtype = np.uint8)
                    rects = np.zeros(img_in.shape, dtype = np.uint8)
                    r_cnt = 0
                    for (x,y,w,h) in detected:
                        arr_img = warped[y:y+h, x:x+w]
                        imf  = np.float32( cv2.resize(arr_img,(32,32)) )
                        imf = (imf-128.)/128.
                        imf = np.expand_dims(imf,-1)

                        #classify
                        o = sess.run([out], feed_dict={inp:[imf], keep_prob:1.0})                        
                        arrow_type = np.argmax(o[0],1)[0]
                        print('type ', arrow_type)
                        
                        r_color = (0,0, 55);

                        if (arrow_type > 0):
                            r_color = (0,0,255);
                            out_type = arrow_type + 100; #separate type range
                            mask[y:y+h, x:x+w] = out_type.astype(np.uint8);

                        if (len(out_dir) > 0):
                            type_str = arrow_types[arrow_type]
                            out_str = '-%d.jpg' % (r_cnt)
                            out_file = os.path.join(out_dir, type_str,
                                                    _file.replace('.jpg',out_str))
                            cv2.imwrite(out_file,arr_img);
                                                            
                        r_cnt = r_cnt + 1

                        cv2.rectangle(rects, (x,y), (x+w,y+h),r_color,5)  
                    
                    rects = cv2.warpPerspective(rects, Minv, warp_shape, 
                                               flags=cv2.INTER_NEAREST)
                    img_in = img_in | rects;
                    cv2.imwrite(o_file,img_in);
                    print(o_file);

                    
                    if (mask.any()):
                        mask = cv2.warpPerspective(mask, Minv, warp_shape, 
                                                   flags=cv2.INTER_LINEAR)
                        mask_path = os.path.join(mask_dir, _file.replace('.jpg','.png'));
                        old_mask_path = os.path.join(mask_dir, _file.replace('.jpg','_.png'));
                                                 
                        old_mask = cv2.imread(mask_path,0);
                        #restore original ROAD_MARKING
                        old_mask[ (old_mask >= 100) & (old_mask < 120)] = ROAD_MARKING
                        
                        idx = np.where( (old_mask==ROAD_MARKING) & (mask > 100) )
                        old_mask[idx] = mask[idx]
                        #print('saving ', mask_path)
                        os.rename(mask_path, old_mask_path)
                        cv2.imwrite(mask_path,old_mask)
            except:
                print("Error: ", sys.exc_info()[0])
                pass


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
        #if (cnt < 2875):
        #    continue;
        img_in = cv2.imread(os.path.join(in_dir, _file),-1)

        
        #undistort!!!
        try:
            und = cv2.undistort(img_in, cam_1_dict['mtx'], cam_1_dict['dist'])
            warped = cv2.warpPerspective(und, M, (img_in.shape[1], img_in.shape[0]), 
                                                     flags=flags)
            
    
    
            cv2.imwrite(os.path.join(out_dir, _file),warped)
        except:
            pass


#%%
if __name__ == "__main__":
    nArg = len(sys.argv)
    args = [None, None, None, None]
    for i in range (1, nArg):
        args[i-1] = sys.argv[i]
    getArrows(args[0], args[1], args[2], args[3])            
