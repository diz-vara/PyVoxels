# -*- coding: utf-8 -*-
"""
Created on Thu May 10 12:03:02 2018

@author: avarfolomeev
"""
import numpy as np
import copy
import struct
import os
from ROS_ts import ROS_ts
from rotationMatrix2Euler import *

import time


point_format = 'fffBBBB'
points_struct_format = "dddiiiB";    
point_keys = ['X', 'Y', 'Z', 'intensity', 'frame', 'ts']    

def unpack(bf):
    x,y,z,s, ns,fr,intensity=struct.unpack(points_struct_format,bf);
    point = dict(zip(point_keys,(x,y,z, 
                                 intensity, fr,
                                 ROS_ts(s,ns))));
    return point;


def read_points(points_file, points_n = -1):
    
    point_size = struct.calcsize(points_struct_format)
    
    points_num = os.path.getsize(points_file)/point_size;

    step = points_num//1000;


    if (points_n > 0 and points_n < points_num):
        points_num = points_n;
        

    pts = []
    cnt = 0;        
    with open(points_file,'rb') as p_file:
        while(cnt < points_num):
            buf = p_file.read(point_size);
            if (len(buf) < point_size):
                break;
            pnt = unpack(buf);
            pts.append(pnt);
            cnt += 1;
            if (cnt%step == 0):
                print(int(cnt*100/points_num),"%")

    return pts;
    
#%%
def read_resulting_points(points_file, points_n = -1):
    
    point_size = struct.calcsize(point_format)
    
    points_num = os.path.getsize(points_file)/point_size;

    if (points_n > 0 and points_n < points_num):
        points_num = points_n;

    step = points_num//1000;

    

    pts = []
    cnt = 0;        
    with open(points_file,'rb') as p_file:
        while(cnt < points_num):
            buf = p_file.read(point_size);
            if (len(buf) < point_size):
                break;
            x,y,z,r,g,b,T = struct.unpack(point_format,buf);
            pnt = np.array([x,y,z]);
            pts.append(pnt);
            cnt += 1;
            if (cnt%step == 0):
                print(int(cnt*100/points_num),"%")

    return np.array(pts);


    
#%%    
    
def translate_points(points_file, imu_to_velo_rot, 
                     IMUdata,
                     base=None, 
                     base_frame=0,
                     time_corr = -0.343):

    start = time.time();
    point_color = np.zeros(4,np.uint8);    

    out = []
    IMU_idx = -1
    IMU_ts = extract_ts(IMUdata)
    Ref_ts = extract_time_ref(IMUdata)
    pos = extract_position(IMUdata);

    time_ref_corr = ROS_ts(time_corr)


    enu = lla2enu(pos,base);
    q = extract_quaternion(IMUdata);
    num_ts = len(IMU_ts)
    times = [];

    enus = []
    frames = []
    qs = []
    point_size = struct.calcsize(points_struct_format)
    
    points_num = os.path.getsize(points_file)/point_size;
    
    path = os.path.split(points_file)[0];
    #frame_path = os.path.join(path,'frames');
    #if os.path.exists(frame_path) is False:
    #    os.makedirs(frame_path)

    ext = '.bin'

    name = 'points_out' + ext;    
    out_path = os.path.join(os.path.split(path)[0], 'out');
    if (not os.path.exists(out_path)):                        
        os.mkdir(out_path);                        
    out_file = os.path.join(out_path,name);
    
    with open(points_file,'rb') as p_file, \
         open(out_file, 'wb') as o_file:
        buf = p_file.read(point_size);
            
        pnt = unpack(buf);

        base_velo_time = pnt['ts'] + time_ref_corr;
    
        print(base_velo_time)
    
        while (IMU_idx < num_ts-3 and base_velo_time >= Ref_ts[IMU_idx+1] ):
            IMU_idx += 1;
        #this is the end    
        if (IMU_idx >= num_ts) :
            return;
    
        print(IMU_idx)
        ## - it's for LONDON!!! IMU_idx -= 2;
        if (IMU_idx < 0):
            IMU_idx = 0;
            
        cnt = 0;
        wrcnt = 0;
            
        
        step = points_num//1000;
        while(1):
            if (True) : #cnt%20 == 0):
                pt = np.array([pnt['X'], pnt['Y'], pnt['Z']])
                
                #if (pt[0] > 0.5) and (pt[0] < 120) and (pt[1] > -25) and (pt[1] < 35) and (pt[2] > -2.8) and (pt[2] < 100.5):
                if (pt[0] > - 5) and (pt[0] < 70) and (pt[1] > -35) and (pt[1] < 35):
                    ts = pnt['ts'] + time_ref_corr;
        
                    #time_ref = ts - base_velo_time + base_time_ref;
                    while (IMU_idx < num_ts-1 and ts > Ref_ts[IMU_idx+1] ):
                        IMU_idx += 1;
                        #this is the end    
                    if (IMU_idx >= num_ts - 1 ) :
                        break;
                
                    dt = (ts - Ref_ts[IMU_idx]).double();
                    time_step = (Ref_ts[IMU_idx+1] - Ref_ts[IMU_idx]).double();
            
                    d_pos = (enu[IMU_idx+1]-enu[IMU_idx])/time_step*dt;   
                    q_int = q[IMU_idx].slerp(q[IMU_idx],q[IMU_idx+1], dt/time_step);                   
        
                    #print(IMU_idx, dt/time_step, q[IMU_idx], q[IMU_idx+1], q_int)                    
                    #break;
                    
                    pt = np.array(pt * imu_to_velo_rot)[0]
                    new_point = np.dot(pt,q_int.rotation_matrix) + enu[IMU_idx] + d_pos;
                    bf = struct.pack(point_format,new_point[0], new_point[1], new_point[2], 
                                     point_color[0],point_color[1],
                                     point_color[2],point_color[3]);
                    o_file.write(bf);
                    wrcnt = wrcnt + 1;
                    
                    #out.append(new_point)
                    #out.append(ned[IMU_idx] + d_pos)
                    #frames.append(pnt['frame']+base_frame);
                    #diffs.append(pnt['frame']);
                    #enus.append(enu[IMU_idx]+d_pos);
                    #qs.append(q_int);
            buf = p_file.read(point_size);
            if (len(buf) < point_size):
                break;
            pnt = unpack(buf);
            cnt += 1;
            if (cnt%step == 0):
                print(cnt*100/points_num,"%")
            
        p_file.close()
        o_file.close()        
                
        #manual zero-az point re-referencing
        #ts = ROS_ts(1520430272,617093234);
        #time_ref = ts - base_velo_time + base_time_ref;
        #IMU_idx = -1;
        #while (time_ref > Ref_ts[IMU_idx+1] and IMU_idx < num_ts):
        #    IMU_idx += 1;
    
        
        #difference from one step, but I;ll use another ?????
        #dt = (time_ref - Ref_ts[IMU_idx]).double();
    
            
        #time_step = (Ref_ts[IMU_idx+1] - Ref_ts[IMU_idx]).double();
        #print (IMU_idx,dt,time_step)        
        
        #d_pos = (enu[IMU_idx+1]-enu[IMU_idx])/time_step*dt + enu[IMU_idx];             
        #q_int = q[IMU_idx].slerp(q[IMU_idx],q[IMU_idx+1], dt/time_step);                   

    
        end = time.time()
        print("Read ", cnt, " points, written ", wrcnt,  " points. Time spend ", end-start, " seconds");
    return cnt#np.array(out),np.array(enus), np.array(frames)-1, np.array(qs)
    #d_pos, q_int
    
#%%
def translate_points_arr(points, imu_to_velo_rot, 
                     IMUdata,
                     base=None, 
                     base_frame=0,
                     time_corr = -0.343):

    point_color = np.zeros(4,np.uint8);    

    out = []
    IMU_idx = -1
    IMU_ts = extract_ts(IMUdata)
    Ref_ts = extract_time_ref(IMUdata)
    pos = extract_position(IMUdata);

    time_ref_corr = ROS_ts(time_corr)


    enu = lla2enu(pos,base);
    q = extract_quaternion(IMUdata);
    num_ts = len(IMU_ts)
    times = [];

    out = []
    enus = []
    frames = []
    qs = []
    
    points_num = len(points)
    
    
    pnt = points[0]

    base_velo_time = pnt['ts'] + time_ref_corr;
    
    print(base_velo_time)
    
    while (IMU_idx < num_ts-3 and base_velo_time >= Ref_ts[IMU_idx+1] ):
        IMU_idx += 1;
    #this is the end    
    if (IMU_idx >= num_ts) :
        return;

    print(IMU_idx)
    ## - it's for LONDON!!! IMU_idx -= 2;
    if (IMU_idx < 0):
        IMU_idx = 0;
        
    base_time_ref = Ref_ts[IMU_idx];    


    cnt = 0;
        
    
    step = points_num//1000;
    
    for pnt in points:
    
        ts = pnt['ts'] + time_ref_corr;
    
            #time_ref = ts - base_velo_time + base_time_ref;
            
        while (IMU_idx < num_ts-1 and ts > Ref_ts[IMU_idx+1] ):
            IMU_idx += 1;
        #this is the end    
        if (IMU_idx >= num_ts - 1 ) :
            break;
                
        dt = (ts - Ref_ts[IMU_idx]).double();
        time_step = (Ref_ts[IMU_idx+1] - Ref_ts[IMU_idx]).double();
        
        d_pos = (enu[IMU_idx+1]-enu[IMU_idx])/time_step*dt;   
        q_int = q[IMU_idx].slerp(q[IMU_idx],q[IMU_idx+1], dt/time_step);                   
        
        pt = np.array([pnt['X'], pnt['Y'], pnt['Z']])
        
        
        #if (pt[0] > 0.5) and (pt[0] < 120) and (pt[1] > -25) and (pt[1] < 35) and (pt[2] > -2.8) and (pt[2] < 100.5):
        if (pt[0] > - 3) and (pt[0] < 70) and (pt[1] > -35) and (pt[1] < 35):
            
            pt = np.array(pt * imu_to_velo_rot)[0]
            new_point = np.dot(pt,q_int.rotation_matrix) + enu[IMU_idx] + d_pos;
            out.append(new_point);                   
            frames.append(pnt['frame']+base_frame);
                #diffs.append(pnt['frame']);
            enus.append(enu[IMU_idx]+d_pos);
            qs.append(q_int);
            cnt += 1;
            if (cnt%step == 0):
                print(cnt*100/points_num,"%")
            
    print(IMU_idx)
                
    return np.array(out),np.array(enus), np.array(frames)-1, np.array(qs)
    #d_pos, q_int

    #%%
    def build_filter (ptr, limits):
        filter_a = ((ptr[:,2] > limits[2,0]) & (ptr[:,1]<limits[1,1]) & (ptr[:,1] > limits[1,0]) & (ptr[:,0] > limits[0,0]) & (ptr[:,0] < limits[0,1]) & (ptr[:,2] < limits[2,1]) )    
        return filter_a
        
    
    