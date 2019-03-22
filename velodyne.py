#!/usr/bin/python

"""
    read or unpack Velodyne VLP-16 data
    usage:
        ./velodyne.py <read | unpack> [bin file dir]
"""

import os
import csv
import sys
import socket
import glob
from datetime import datetime, timedelta
import struct
import time
import traceback
import numpy as np
from multiprocessing import Process, Queue, Pool


import VLP16defs
from ROS_ts import ROS_ts
import copy

from IMUdata import *


import pickle
#%%
def read_ts(file):
    ts = []
    with open(file, 'rt') as ts_f:
        for line in ts_f:
            t = line.split()[0].split('.')
            ts_s = int(t[0])
            ts_ns = int(t[1])
            ts.append( ROS_ts(ts_s, ts_ns))
    return np.array(ts)


#%%

point_keys = ['X', 'Y', 'Z', 'intensity', 'frame', 'ts']    
    
def read_velo_file(path,framesNr=1e9):
    
    format_out = 'fffBBBB'

    
    packet_cnt = 0;
    ts_file = os.path.join(path,'timestampsU.txt')
    time_stamps = read_ts(ts_file)
    
    v_file = os.path.join(path, 'velodyne.bin')
    o_file = os.path.join(path, 'points__.bin');

    

    points = []
    frame_stamps = []
    frame = 0
    
    base_ts = ROS_ts(0)
    prev_ts = 0
    corr_ts = 0
    
    with open(v_file, 'rb') as vf, \
         open(o_file,'wb') as out_file:
        d_az = 0.41111;  #default az step    
        old_az = -1;
        prev_az = 181;
        cnt = 0;
        
        while (1):
            #scan = np.fromfile(vf, dtype=np.uint8, count=PACKET_SIZE);
            scan = vf.read(PACKET_SIZE);
            if (len(scan) < PACKET_SIZE ):
                break;

            ts=struct.unpack_from('<I', scan, TS_OFFSET)[0]
                
            if (prev_ts == 0):
                prev_ts = ts;
                corr_ts = 0;
                continue; #skip first block
                
            #corrections for the case of GPS error 
            # that causes extra second added 
            ts -= corr_ts;  
            #print(ts-prev_ts)
            if (ts - prev_ts > 1000000):
                ts -= 1000000;
                corr_ts += 1000000;
            prev_ts = ts;    
            
                
            
            if (base_ts == 0):
                base_ts = time_stamps[0] - ts * 1000;

            firing_ts = base_ts + ts * 1000;
            #print (firing_ts, time_stamps[packet_cnt], firing_ts-time_stamps[packet_cnt])

            for block in range(BLOCKS_PER_PACKET):
                offset = block * SIZE_BLOCK;
                head=struct.unpack_from('<HH',scan, offset)     
                az = head[1]/100.;
                #print("AZ = {:f}, dAZ = {:f}".format(az,d_az) )
                #todo - interpolate az!!!
                offset += 4;
                if (old_az >= 0):
                    d_az = az-old_az;
                old_az = az;
                if (d_az < 0):
                    d_az += 360;
                    
                if (az > 180):
                    az -= 360;

                dd_az = d_az/VLP16_BLOCK_TDURATION * VLP16_DSR_TOFFSET;
                
                for fire in range(VLP16_FIRINGS_PER_BLOCK):
                    arr = struct.unpack_from('<'+'HB'*VLP16_SCANS_PER_FIRING,
                                             scan,offset);
                    #print(arr)
                    point_az = az;
                    point_ts = copy.copy(firing_ts);
                    for  laser_id in range (VLP16_SCANS_PER_FIRING):
                        R = arr[laser_id*2] * DISTANCE_RESOLUTION;
                        intensity = arr[laser_id*2+1];
                        if ( ( (R > 1 and (point_az < -10 or point_az > 10)) or R > 2.7 )  and intensity > 0):
                            alpha = 0 - point_az * np.pi/180. ;  #NEGATE!!!  
                            omega = LASER_ANGLES[laser_id] * np.pi / 180.0
                            X = R * np.cos(omega) * np.cos(alpha)
                            Y = R * np.cos(omega) * np.sin(alpha)
                            Z = R * np.sin(omega)  #do not add ned[2] - alt is wrong
    
                            point = dict(zip(point_keys,(X, Y, Z, 
                                                         intensity, frame,
                                                         copy.copy(point_ts))))
                            #pickle.dump(point,p_file);
                            bf = pack(point)
                            out_file.write(bf);
                            cnt += 1
                            points.append(point)
                        point_az += dd_az;
                        if (point_az > 180):
                            point_az -= 360;

                        if (point_az < -180):
                            point_az += 360;
                        if (point_az - prev_az < 0):
                            frame += 1;
                        if (prev_az < 0 and point_az >= 0):
                            frame_stamps.append( point_ts) ;
                        prev_az = point_az    
                        point_ts += VLP16_DSR_TOFFSET_NS;
                        #print(point_ts, point['ts'], points[-1]['ts'])
                        #if (len(points)> 1):
                        #    print(points[-2]['ts'])
                       
                    offset = offset + VLP16_SCANS_PER_FIRING*RAW_SCAN_SIZE;
                    az += d_az/2;
                    if (az > 180):
                        az -= 360;
                    if (az < -180):
                        az += 360;
                    firing_ts += VLP16_FIRING_TOFFSET_NS


            #tail=struct.unpack_from('<IH', scan, offset)
            #print("{:d} {:d} {:X}".
            #      format(offset,tail[0], tail[1]))
            if (frame >= framesNr):
                break;
            packet_cnt = packet_cnt + 1
            print(packet_cnt)
            #if (packet_cnt > 4000):
            #    break;
    out_file.close();        
    return points
            
    

#%%

points_struct_format = "dddiiiB";

def pack(p0):
    bf=struct.pack(points_struct_format,p0['X'],p0['Y'],p0['Z'], #3 x double
                   p0['ts'].s, p0['ts'].ns, p0['frame'],          #2 x int32 
                   p0['intensity'])                   #Byte
    return bf    
    

def unpack(bf):
    x,y,z,s, ns,fr,intensity=struct.unpack(points_struct_format,bf);
    point = dict(zip(point_keys,(x,y,z, 
                                 intensity, fr,
                                 ROS_ts(s,ns))));
    return point;


#%%

def save_csv(path, data):
    with open(path, 'w') as fp:
        wr = csv.writer(fp, delimiter=',')
        wr.writerows(data)

def calc(dis, azimuth, laser_id, timestamp):
    R = dis * DISTANCE_RESOLUTION
    omega = LASER_ANGLES[laser_id] * np.pi / 180.0
    alpha = azimuth / 100.0 * np.pi / 180.0
    X = R * np.cos(omega) * np.sin(alpha)
    Y = R * np.cos(omega) * np.cos(alpha)
    Z = R * np.sin(omega)
    return [X, Y, Z, timestamp]


#%%                            
def save_package(dirs, data_queue):
    try:
        if os.path.exists(dirs) is False:
            os.makedirs(dirs)
        cnt = 0
        fp = None
        while True:
            if data_queue.empty():
                pass
            else:
                msg = data_queue.get()
                data = msg['data']
                ts = msg['time']
                print (ts, len(data), 'queue size: ', data_queue.qsize(), cnt)
                if fp == None or cnt == 1000000:
                    if fp != None:
                        fp.close()
                    file_fmt = os.path.join(dirs, '%Y-%m-%d_%H%M')
                    path = str(datetime.now().strftime(file_fmt)) + '.bin'
                    logger.info('save to' + path)
                    print ('save to ', path)
                    fp = open(path, 'ab')
                    cnt = 0
                cnt += 1
                fp.write('%.6f' % ts)
                fp.write(data)
    except KeyboardInterrupt as e:
        print(e)
    finally:
        if fp != None:
            fp.close()

def capture(port, data_queue):
    soc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    soc.bind(('', port))
    try:
        while True:
            try:
                data = soc.recv(2000)
                if len(data) > 0:
                    assert len(data) == 1206, len(data)
                    data_queue.put({'data': data, 'time': time.time()})
            except Exception as e:
                print( dir(e), e.message, e.__class__.__name__)
                traceback.print_exc(e)
    except KeyboardInterrupt as e:
        print (e)


if __name__ == "__main__":
    #points21_f, frames21_f=read_velo_file('E:\\Data\\Voxels\\2018_03_08\\L21\\velodyne_packets\\')
    pointsR1_11 = read_velo_file('F://Data//Voxels/20180525/ride01-11/velodyne_packets/')
    

"""    
    if len(sys.argv) < 3:
        print (__doc__)
        sys.exit(2)
    if sys.argv[1] == 'read':
        top_dir = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        processA = Process(target = capture, args = (PORT, DATA_QUEUE))
        processA.start()
        processB = Process(target = save_package, args = (sys.argv[2] + '/' + top_dir, DATA_QUEUE))
        processB.start()
    else:
        unpack(sys.argv[2])
"""
