# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 22:54:00 2017

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
import tensorflow as tf

from voxTexts.Classifier import Classifier

from read_frozen import read_frozen

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
# dist_pickle = pickle.load( open( "mtx_dist.p", "rb" ) )
# mtx = dist_pickle["mtx"]
# dist = dist_pickle["dist"]

thread_local_data = threading.local()


class CamParams:
    def __init__(self, calibration_file):
        self.calibration = pickle.load(open(calibration_file, 'rb'))


class Cam0Params(CamParams):
    def __init__(self, calibration_file):
        super().__init__(calibration_file)

        src = np.float32(
            [
                [796, 726],
                [1118, 776],
                [864, 921],
                [514, 823]
            ]
        )

        origin = (src[3][0] + 90, src[3][1] + 80)
        ratio = 1.2
        width = 250
        height = width / ratio

        dst = np.float32(
            [
                [origin[0], origin[1] - height],
                [origin[0] + width, origin[1] - height],
                [origin[0] + width, origin[1]],
                [origin[0], origin[1]]
            ])

        self.perspectiveTransform = cv2.getPerspectiveTransform(src, dst)
        self.inversePerspectiveTransform = cv2.getPerspectiveTransform(dst, src)


class Cam1Params(CamParams):
    def __init__(self, calibration_file):
        super().__init__(calibration_file)

        src = np.float32(
            [[337, 518],
             [334, 831],
             [770, 515],
             [805, 824]])

        left = 10
        top = 25
        dst = np.float32(
            [[left - 4.27 - 2.7, top - 17.08],
             [left - 1.34, top - 5.08],
             [left + 2.75, top - 17.58],
             [left + 1.05, top - 5.2]])

        self.perspectiveTransform = cv2.getPerspectiveTransform(src, dst * 50)
        self.inversePerspectiveTransform = cv2.getPerspectiveTransform(dst * 50, src)


class Cam2Params(CamParams):
    def __init__(self, calibration_file):
        super().__init__(calibration_file)

        src = np.float32(
            [
                [478, 711],
                [751, 803],
                [450, 880],
                [194, 746]
            ]
        )

        origin = (220, src[2][1] + 20)
        ratio = 1.2 #0.8333
        width = 250
        height = width / ratio

        dst = np.float32(
            [
                [origin[0] + width, origin[1] - height],
                [origin[0] + width, origin[1]],
                [origin[0], origin[1]],
                [origin[0], origin[1] - height]
            ])

        self.perspectiveTransform = cv2.getPerspectiveTransform(src, dst)
        self.inversePerspectiveTransform = cv2.getPerspectiveTransform(dst, src)


# e) use cv2.warpPerspective() to warp your image to a top-down view
# image0 = cv2.imread('E:\\Data\\Voxels\\201902_USA\\20190216_151511_\\argus_cam_1\\data\\1550358975.579326630.jpg')
# image = cv2.undistort(image0, mtx, dist, None, mtx)
# warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
# plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

GetArrowsAndTextsParams = namedtuple('GetArrowsAndTextsParams',
                                     ('arrowsCascadeClassifierPath', 'arrowsNet', 'inp', 'keep_prob', 'out', 'arrow_types')
                                     ('textsCascadeClassifierPath', 'textsNet'))

GetArrowsAndTextsResult = namedtuple('GetArrowsAndTextsResult',
                                     ('detected_arrows_count, non_zero_arrows_count')
                                     ('detected_texts_count, non_zero_texts_count'))

# %%
ROAD_MARKING = 7


def doGetArrows(img_in,
                _file,
                out_dir,
                params: GetArrowsAndTextsParams,
                warped: bool,
                sess: tf.Session,
                inout_rects):
    try:
        cascadeArrows = thread_local_data.cascadeArrows
    except AttributeError:
        cascadeArrows = cv2.CascadeClassifier(params.arrowsCascadeClassifierPath)
        thread_local_data.cascadeArrows = cascadeArrows


    detected = cascadeArrows.detectMultiScale(warped,
                                              1.15,
                                              minNeighbors=1,
                                              minSize=(30, 30),
                                              maxSize=(400, 400))
    count = 0

    if len(detected) > 0:
        # print("Detected ", len(detected), " arrows")
        mask = np.zeros(img_in.shape[:2], dtype=np.uint8)
        rects = np.zeros(img_in.shape, dtype=np.uint8)
        r_cnt = 0
        for (x, y, w, h) in detected:
            arr_img = warped[y:y + h, x:x + w]
            imf = np.float32(cv2.resize(arr_img, (32, 32)))
            imf = (imf - 128.) / 128.
            imf = np.expand_dims(imf, -1)

            # classify
            o = sess.run([params.out], feed_dict={params.inp: [imf], params.keep_prob: 1.0})
            arrow_type = np.argmax(o[0], 1)[0]
            # print('arrow type ', arrow_type)

            r_color = (0, 0, 55)

            if arrow_type > 0:
                count += 1
                r_color = (0, 0, 255)
                out_type = arrow_type + 100  # separate type range
                mask[y:y + h, x:x + w] = out_type.astype(np.uint8)

            # if len(out_dir) > 0:
            #     type_str = params.arrow_types[arrow_type]
            #     out_str = '-%d.jpg' % (r_cnt)
            #     out_file = os.path.join(out_dir,
            #                             type_str,
            #                             _file.replace('.jpg', out_str))
            #     cv2.imwrite(out_file, arr_img)

            r_cnt = r_cnt + 1

            cv2.rectangle(rects, (x, y), (x + w, y + h), r_color, 5)

        inout_rects = inout_rects | rects

    return GetArrowsAndTextsResult(detected_arrows_count=len(detected),
                                   non_zero_arrows_count=count)


def doGetTexts(img_in,
               out_dir,
               _file,
               params: GetArrowsAndTextsParams,
               warped: bool,
               texts_net: Classifier.Classifier,
               inout_rects):
    try:
        cascadeTexts = thread_local_data.cascadeTexts
    except AttributeError:
        cascadeTexts = cv2.CascadeClassifier(params.textsCascadeClassifierPath)
        thread_local_data.cascadeTexts = cascadeTexts

    detected = cascadeTexts.detectMultiScale(warped,
                                             1.1,
                                             minNeighbors=1,
                                             minSize=(50, 75),
                                             maxSize=(300, 450))

    count = 0

    if len(detected) > 0:
        # print("Detected ", len(detected), " texts")
        mask = np.zeros(img_in.shape[:2], dtype=np.uint8)
        rects = np.zeros(img_in.shape, dtype=np.uint8)
        r_cnt = 0
        for (x, y, w, h) in detected:
            img = warped[y:y + h, x:x + w]
            img = np.expand_dims(img, -1)

            # classify
            class_id = texts_net.classify_image_from_array(img)
            # print('text type ', class_id)

            r_color = (0, 100, 0)

            if class_id > 0:
                count += 1
                r_color = (0, 255, 0)
                out_type = class_id + 200  # separate type range
                mask[y:y + h, x:x + w] = out_type.astype(np.uint8)

            # if len(out_dir) > 0:
            #     type_str = params.arrow_types[texts_net.get_class_name_by_id(class_id)]
            #     out_str = '-%d.jpg' % (r_cnt)
            #     out_file = os.path.join(out_dir, type_str,
            #                             _file.replace('.jpg', out_str))
            #     cv2.imwrite(out_file, img)

            r_cnt = r_cnt + 1

            cv2.rectangle(rects, (x, y), (x + w, y + h), r_color, 5)

        inout_rects = inout_rects | rects

    return GetArrowsAndTextsResult(detected_texts_count=len(detected),
                                   non_zero_texts_count=count)


def doGetArrowsAndTexts(_file,
                        out_dir,
                        camera_params: CamParams,
                        params: GetArrowsAndTextsParams,
                        sess: tf.Session):
    img_in = cv2.imread(_file, -1)  # GRAY only!!!
    grey = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    warp_shape = (img_in.shape[1], img_in.shape[0])
    # if warp_shape_0 != warp_shape:
    #    umap = cv2.initUndistortRectifyMap(camera_params.calibration['mtx'],
    #                                       camera_params.calibration['dist'],
    #                                       None,
    #                                       camera_params.calibration['mtx'],
    #                                       warp_shape,
    #                                       cv2.CV_32FC1)
    #    warp_shape_0 = warp_shape
    directory, file = os.path.split(_file)
    o_file = os.path.join(out_dir, file)
    #### o_file = _file.replace('.jpg', '_.jpg')
    _file = os.path.split(_file)[-1]
    # if (True):

    try:
        # und = cv2.undistort(grey, camera_params.calibration['mtx'], camera_params.calibration['dist'])
        # warped = cv2.warpPerspective(und,
        #                              camera_params.perspectiveTransform,
        #                              warp_shape,
        #                              flags=cv2.INTER_LINEAR)
        warped = grey

        rects = np.zeros(img_in.shape, dtype=np.uint8)

        arrows_result = doGetArrows(img_in, params, warped, sess, rects)

        texts_result = doGetTexts(img_in, params, warped, params.texts_net, rects)

        result = GetArrowsAndTextsResult(detected_arrows_count=arrows_result.detected_arrows_count,
                                         non_zero_arrows_count=arrows_result.non_zero_arrows_count,
                                         detected_texts_count=texts_result.detected_texts_count,
                                         non_zero_texts_count=texts_result.non_zero_texts_count)

        # rects = cv2.warpPerspective(rects,
        #                             camera_params.inversePerspectiveTransform,
        #                             warp_shape,
        #                             flags=cv2.INTER_NEAREST)
        img_in = img_in | rects
        cv2.imwrite(o_file, img_in)
        # print(o_file)

        # if mask.any():
        #     mask = cv2.warpPerspective(mask,
        #                                camera_params.inversePerspectiveTransform,
        #                                warp_shape,
        #                                flags=cv2.INTER_LINEAR)
        #     mask_path = os.path.join(mask_dir, _file.replace('.jpg', '.png'))
        #     old_mask_path = os.path.join(mask_dir, _file.replace('.jpg', '_.png'))
        #
        #     old_mask = cv2.imread(mask_path, 0)
        #     # restore original ROAD_MARKING
        #     old_mask[(old_mask >= 100) & (old_mask < 120)] = ROAD_MARKING
        #
        #     idx = np.where((old_mask == ROAD_MARKING) & (mask > 100))
        #     old_mask[idx] = mask[idx]
        #     # print('saving ', mask_path)
        #     os.rename(mask_path, old_mask_path)
        #     cv2.imwrite(mask_path, old_mask)

        return result

    except Exception as e:
        print("Error:", e)
        return None


# FIXME start = 34600
def getArrowsAndTexts(params: GetArrowsAndTextsParams, camera_params, in_dir, mask_dir, out_dir="", start=0, end=-1):
    print('Loading images from ' + in_dir)

    if (out_dir is None):
        out_dir = ""
    if (len(out_dir) > 0):
        try:
            os.makedirs(out_dir)
        except:
            pass

        arrows_dir = os.path.join(out_dir, 'arrows')
        for sub_dir in params.arrow_types:
            try:
                os.makedirs(os.path.join(arrows_dir, sub_dir))
            except:
                pass;
        print(" writing arrows to " + arrows_dir)

        texts_dir = os.path.join(out_dir, 'texts')
        for sub_dir in params.textsNet.get_class_names():
            try:
                os.makedirs(os.path.join(texts_dir, sub_dir))
            except:
                pass;
        print(" writing texts to " + texts_dir)


    for f in os.listdir(in_dir):
        name, ext = os.path.splitext(f)
        if name.endswith('_'):
            os.remove(os.path.join(in_dir, f))

    im_files = sorted(glob.glob(in_dir + '/*.*g'))

    if (start is None):
        start = 0

    if (end is None or end < 0):
        end = len(im_files)
    im_files = im_files[start:end + 1]
    config = tf.ConfigProto()
    # config.gpu_options.visible_device_list = '1'
    # warp_shape_0 = (0, 0)

    futures = set()

    start_time = time.time()

    with tf.Session(graph=params.arrowsNet, config=config) as sess:
        with ThreadPoolExecutor() as executor:
            for _file in im_files:
                future = executor.submit(doGetArrowsAndTexts, _file, out_dir, camera_params, params, sess)
                futures.add(future)

            total_count = len(futures)
            detected_count = 0
            non_zero_count = 0
            while futures:
                done, futures = concurrent.futures.wait(futures, timeout=1)
                if done:
                    for future in done:
                        result = future.result()
                        detected_count += result.detected_count
                        non_zero_count += result.non_zero_count
                    print('> Processed {} of {}, detected {} arrows, non-zero: {}'.format(
                        total_count - len(futures),
                        total_count,
                        detected_count,
                        non_zero_count))

    time_spent = time.time() - start_time
    print('Time spent: {:.03f} seconds'.format(time_spent))


CAMERA_PARAMS = (
    Cam0Params,
    Cam1Params,
    Cam2Params
)


def do_unpersp(camera_params, img_path, out_path, flags):
    try:
        img_in = cv2.imread(img_path, -1)
        undistored = cv2.undistort(img_in, camera_params.calibration['mtx'], camera_params.calibration['dist'])
        height, width = img_in.shape[:2]
        warped = cv2.warpPerspective(undistored, camera_params.perspectiveTransform, (width, height), flags=flags)
        cv2.imwrite(out_path, warped)
    except Exception as e:
        print('Failed to process {}: {}'.format(img_path, e))


# %%
def unpersp(in_dir, out_dir, _flag, camera_params):
    """Load road masks from file.
    Images are RGB, with:
        (255,0,255) for road
        (255,0,0) for not_road
    """

    if (_flag is None or _flag == 0):
        flags = cv2.INTER_LINEAR
    else:
        _flag = 1
        flags = cv2.INTER_NEAREST

    try:
        os.makedirs(out_dir)
    except:
        pass

    print('Loading images from ' + in_dir + " writing to " + out_dir)

    im_files = sorted(os.listdir(in_dir))

    futures = set()

    start_time = time.time()

    cnt = 0
    with ThreadPoolExecutor() as executor:
        for _file in im_files:
            cnt += 1

            img_path = os.path.join(in_dir, _file)
            out_path = os.path.join(out_dir, _file)

            future = executor.submit(do_unpersp, camera_params, img_path, out_path, flags)
            futures.add(future)

        total_count = len(futures)
        while futures:
            done, futures = concurrent.futures.wait(futures, timeout=1)
            if done:
                print('> Processed {} of {}'.format(total_count - len(futures), total_count))

    time_spent = time.time() - start_time
    print('Time spent: {:.03f} seconds'.format(time_spent))




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
    parameters_dir = '/media/nvidia/Data/Voxels/Arrows/'

    camera_params0 = CAMERA_PARAMS[0](parameters_dir + 'sony_cam_0_dict.p')
    camera_params1 = CAMERA_PARAMS[1](parameters_dir + 'sony_cam_1_dict.p')
    camera_params2 = CAMERA_PARAMS[2](parameters_dir + 'sony_cam_2_dict.p')

    # arrowsCascadeClassifierPath = cv2.CascadeClassifier(args.cascade)
    arrowsCascadeClassifierPath = parameters_dir + 'cArr_24_1_50_s11.xml'

    # arrowsNet = read_frozen(args.arrows_net)
    arrowsNet = read_frozen(parameters_dir +  'arrows-5.frozen_model.pb')

    textsCascadeClassifierPath = parameters_dir + 'cTxt_16x24_1_60_s06.xml'

    textsNet = Classifier.Classifier(model_file_path=parameters_dir + 'texts.model.h5',
                                     category_to_id_mapping_file_path='')

    inp = arrowsNet.get_tensor_by_name('input_image:0')
    keep_prob = arrowsNet.get_tensor_by_name('keep_prob:0')
    out = arrowsNet.get_tensor_by_name('lin2:0')
    arrow_types = ["BG", "LU", "LUR", "RU", "U", "dL", "dR", "diagL", "diagR", "uL", "uR"]

    get_arrows_params = GetArrowsAndTextsParams(arrowsCascadeClassifierPath=arrowsCascadeClassifierPath,
                                                arrowsNet=arrowsNet,
                                                inp=inp,
                                                keep_prob=keep_prob,
                                                out=out,
                                                arrow_types=arrow_types,
                                                textsCascadeClassifierPath=textsCascadeClassifierPath,
                                                textsNet=textsNet)

    ############################################################################

    # getArrows(get_arrows_params,
    #          camera_params,
    #          'C:\\src\\argus_cam_data\\argus_cam_0\\data_warp',
    #          'C:\\src\\argus_cam_data\\argus_cam_0\\Xroad',
    #          'C:\\src\\argus_cam_data\\argus_cam_0\\data_warp_out')

    # getArrows(get_arrows_params,
    #          camera_params,
    #          'C:\\src\\argus_cam_data\\argus_cam_0\\suspicious_orig',
    #          'C:\\src\\argus_cam_data\\argus_cam_0\\Xroad',
    #          'C:\\src\\argus_cam_data\\argus_cam_0\\suspicious_update')

    # getArrows(get_arrows_params, camera_params, 'D:\\argus_cam_data\\argus_cam_0', 'D:\\argus_cam_data\\argus_cam_0\\out_001_arr')
    # getArrows(get_arrows_params, camera_params, 'D:\\argus_cam_data\\argus_cam_1', 'D:\\argus_cam_data\\argus_cam_1\\out_111_arr')
    # getArrows(get_arrows_params, camera_params, 'D:\\argus_cam_data\\argus_cam_2', 'D:\\argus_cam_data\\argus_cam_2\\out_221_arr')

    # getArrows(get_arrows_params, camera_params, 'D:\\argus_cam_data\\arr-test\\left', 'D:\\argus_cam_data\\arr-test\\left\\out_001_arr')
    # getArrows(get_arrows_params, camera_params, 'D:\\argus_cam_data\\arr-test\\front', 'D:\\argus_cam_data\\arr-test\\front\\out_001_arr')
    # getArrows(get_arrows_params, camera_params, 'D:\\argus_cam_data\\arr-test\\right', 'D:\\argus_cam_data\\arr-test\\right\\out_001_arr')

    # unpersp(args.input_dir, args.output_dir, None, camera_params)
    # unpersp('D:\\argus_cam_data\\argus_cam_0\\data',
    #        'C:\\src\\argus_cam_data\\argus_cam_0\\data_warp',
    #        None,
    #        camera_params)
    # unpersp('D:\\argus_cam_data\\argus_cam_1\\data',
    #        'D:\\argus_cam_data\\argus_cam_1\\data_warp',
    #        None,
    #        camera_params)
    # unpersp('D:\\argus_cam_data\\argus_cam_2\\data',
    #        'D:\\argus_cam_data\\argus_cam_2\\data_warp',
    #        None,
    #        camera_params)

    # unpersp('D:\\argus_cam_data\\argus_cam_0', 'in_001', 'out_001', None, camera_params)
    # unpersp('D:\\argus_cam_data\\argus_cam_1', 'in_001', 'out_001', None, camera_params)
    # unpersp('D:\\argus_cam_data\\argus_cam_2', 'in_001', 'out_001', None, camera_params)
    # unpersp('E:\\unpersp', 'in', 'out', None, camera_params)
