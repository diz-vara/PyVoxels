# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 22:54:00 2019

@author: Anton Varfolomeev
"""
import concurrent
import glob
import os
import pickle
import threading
import time
from collections import namedtuple
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor

import cv2
import numpy as np

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
# dist_pickle = pickle.load( open( "mtx_dist.p", "rb" ) )
# mtx = dist_pickle["mtx"]
# dist = dist_pickle["dist"]

thread_local_data = threading.local()


LICENSE_PLATE = 24
PERSON = 23


def doBlur(_file, mask_dir, out_dir):
    try:
        img_in = cv2.imread(_file, -1)  
        directory, file = os.path.split(_file)
     
        o_file = os.path.join(out_dir, file)
        #### o_file = _file.replace('.jpg', '_.jpg')
        _file = os.path.split(_file)[-1]
    
    
        mask_path = os.path.join(mask_dir, _file.replace('.jpg', '.png'))
        mask = cv2.imread(mask_path, 0)
         
        idx = np.where((mask == LICENSE_PLATE) | (mask == PERSON))
        img_blur = cv2.blur(img_in,(7,7))
        img_in[idx] = img_blur[idx]
        cv2.imwrite(o_file, img_in)

     
    except Exception as e:
        print("Error:", e)
        return None


# FIXME start = 34600
def blurNumbers(in_dir, mask_dir, out_dir="", start=0, end=-1):
    print('Loading images from ' + in_dir)

    if (out_dir is None):
        out_dir = ""
    if (len(out_dir) > 0):
        try:
            os.makedirs(out_dir)
        except:
            pass


        print(" writing blurred images to " + out_dir)


    im_files = sorted(glob.glob(in_dir + '/*.*g'))

    if (start is None):
        start = 0

    if (end is None or end < 0):
        end = len(im_files)
    im_files = im_files[start:end + 1]

    futures = set()

    start_time = time.time()

    with ThreadPoolExecutor() as executor:
        for _file in im_files:
            future = executor.submit(doBlur, _file, mask_dir, out_dir)
            futures.add(future)

        total_count = len(futures)
        detected_count = 0
        non_zero_count = 0
        print('Processing ', total_count, ' files')
        while futures:
            done, futures = concurrent.futures.wait(futures, timeout=1)
            if (done):
                print('> Processed {} of {}'.format(total_count - len(futures), total_count))


    time_spent = time.time() - start_time
    print('Time spent: {:.03f} seconds'.format(time_spent))





# %%

# %%
if __name__ == "__main__":
    ### parser = argparse.ArgumentParser()

    ### parser.add_argument('--camera-index', type=int, required=True)
    ### parser.add_argument('--calibration', required=True)
    ### parser.add_argument('--cascade', required=True)
    ### parser.add_argument('--arrows-net', required=True)
    ### parser.add_argument('input_dir', help='Input directory')
    ### parser.add_argument('output_dir', help='Output directory')

    ### args = parser.parse_args()

    ### print('Camera index:', args.camera_index)
    ### print('Calibration file:', args.calibration)
    ### print('Cascade file:', args.cascade)
    ### print('Arrows net file: ', args.arrows_net)
    ### print('Input directory:', args.input_dir)
    ### print('Output directory:', args.output_dir)

    # camera_params = CAMERA_PARAMS[args.camera_index](args.calibration)

    #parameters_dir = 'C:\\src\\GIT\\PyVoxels\\'

    ############################################################################

    blurNumbers('E:\\Data\\Voxels\\201809_usa\\test7_2\\argus_cam_2\\data\\',
                'E:\\Data\\Voxels\\201809_usa\\test7_2\\argus_cam_2\\Xroad\\',
                'E:\\Data\\Voxels\\201809_usa\\test7_2\\argus_cam_2\\blurred\\')
    
