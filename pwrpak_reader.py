# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 09:16:15 2018

@author: avarfolomeev
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 
import cv2
from pyquaternion import Quaternion
import datetime, calendar
import navpy

#degrees minutes.minutes to degrees
def dm2degrees(dm):
        deg = int (dm/100)
        minutes = dm - deg*100
        degrees = deg + minutes/60.;
        return degrees
        
#degrees minutes.minutes to degrees
def hms2seconds(hms):
        h = int (hms/10000)
        minutes = int ((hms - h * 10000)/100)
        seconds = hms - h*10000 - minutes * 100;
        seconds = seconds + (h * 60 + minutes) * 60 
        return seconds
    

def week_seconds_to_utc(gpsweek,gpsseconds,leapseconds):
    import datetime, calendar
    datetimeformat = "%Y-%m-%d %H:%M:%S"
    epoch = datetime.datetime.strptime("1980-01-06 00:00:00",datetimeformat)
    elapsed = datetime.timedelta(days=(gpsweek*7),seconds=(gpsseconds+leapseconds))
    return (epoch+elapsed).timestamp()

def read_pwrpak_log(fname):
    
    point0 = {'time':0., 'lat':0., 'lon':0., 'alt_a':0., 'alt_g':0., 'quality':0.,
             'bl_north':0., 'bl_east':0., 'bl_up':0, 'bl_length':0., 'bl_course':0., 'bl_pitch':0.,
             'year':0, 'month':0, 'day':0,
             'rmc_course':-1., 'true_course':-1.,
             'imu_roll':0., 'imu_pitch':0., 'imu_yaw':0.}
             
    imu0 = {'time':0, 'accX':0, 'accY':0, 'accZ':0,
               'gyrX':0, 'gyrY':0, 'gyrZ':0}
    imu = []           
    points = []
    point = point0.copy();
    cnt = np.zeros(20)
    with open (fname) as csvfile:
        pclreader = csv.reader(csvfile, delimiter = ',', skipinitialspace=True)
        for row in pclreader:
            if ( '$--GPGGA' in row):
                #process GGA sentence
                time = hms2seconds(float(row[1]))
                if (time != point['time']):
                    if (point['quality'] != 0):
                        points.append(point)
                    point = point0.copy();
                    point['time'] = time;
                
                lat = dm2degrees(float(row[2]))
                if (row[3] != 'N'):
                    lat = -lat;
                point['lat'] = lat   
                lon = dm2degrees(float(row[4]))
                if (row[5] != 'E'):
                    lon = -lon    
                point['lon'] = lon    
                point['alt_a'] = float(row[9])
                point['alt_g'] = float(row[11])
                point['quality'] = int(row[6])
                print (row[0], row[1])
            elif  ('$GPRMC' in row):
                time = hms2seconds(float(row[1]))
                if (time != point['time']):
                    if (point['quality'] != 0):
                        points.append(point)
                    point = point0.copy();
                    point['time'] = time;
                point['rmc_course'] = float(row[8])
                print (row[0], row[1])
                
            elif  ('$GPHDT' in row):
                #process GPHDT 
                if (row[1]):
                    point['true_course'] = float(row[1])
            elif  ('$GPVTG' in row):
                #process GPVTG
                print (row[0], row[1])
            elif  ('$GPZDA' in row):
                #process GPZDA
                time = hms2seconds(float(row[1]))
                if (time != point['time']):
                    if (point['quality'] != 0):
                        points.append(point)
                    point = point0.copy();
                    point['time'] = time;
                point['day'] = int(row[2])
                point['month'] = int (row[3])
                point['year'] = int (row[4])
                print (row[0], row[1])
            elif  ('$PNVGIMU' in row):
                #process PNVGIMU
                time = hms2seconds(float(row[1]))
                if (time != point['time']):
                    if (point['quality'] != 0):
                        points.append(point)
                    point = point0.copy();
                    point['time'] = time;

                point['imu_roll'] = float(row[2])
                point['imu_pitch'] = float(row[3])
                point['imu_yaw'] = float(row[4])                
                print (row[0], row[1])
            elif  ('$PNVGBLS' in row):
                #process PNVGBLS

                time = hms2seconds(float(row[1]))
                if (time != point['time']):
                    if (point['quality'] != 0):
                        points.append(point)
                    point = point0.copy();
                    point['time'] = time;

                if (row[2]):
                    point['bl_north'] = float(row[2])
                if (row[3]):
                    point['bl_east'] = float(row[3])
                if (row[4]):
                    point['bl_up'] = float(row[4])
                if (row[5]):
                    point['bl_length'] = float(row[5])
                if (row[6]):
                    point['bl_course'] = float(row[6])
                if (row[7]):
                    point['bl_pitch'] = float(row[7])
                print (row[0], row[1])
            elif ('<INSPVA' in row):
                point = point0.copy()
                #first string is a header, skip it
                row = next(pclreader)
                # week seconds lat lon alt Nvel Evel Uvel Roll Pith Azim 
                week = int(row[1])
                seconds = float(row[2])
                ts = week_seconds_to_utc(week,seconds,0)                
                print (ts)
                # LEAP SECONDS !!!!
                point['time'] = ts
                point['lat'] = float(row[3])
                point['lon'] = float(row[4])
                point['alt_a'] = float(row[5])
                point['imu_roll'] = float(row[9])
                point['imu_pitch'] = float(row[10])
                point['imu_yaw'] = float(row[11])
                points.append(point)
            elif ('#INSPVAXA' in row):
                point = point0.copy()
                # week seconds lat lon alt Nvel Evel Uvel Roll Pith Azim 
                week = int(row[5])
                seconds = float(row[6])
                ts = week_seconds_to_utc(week,seconds,0)                
                #print (ts)
                # LEAP SECONDS !!!!
                point['time'] = ts
                point['lat'] = float(row[11])
                point['lon'] = float(row[12])
                point['alt_a'] = float(row[13])
                point['alt_t'] = float(row[14])

                point['imu_roll'] = float(row[18])
                point['imu_pitch'] = float(row[19])
                point['imu_yaw'] = float(row[20])
                points.append(point)
                cnt[0] += 1
            elif ('%RAWIMUSXA' in row or '[COM1]%RAWIMUSXA' in row):
                point = point0.copy()
                # week seconds lat lon alt Nvel Evel Uvel Roll Pith Azim 
                week = int(row[4])
                seconds = float(row[5])#.split(';')[0])
                ts = week_seconds_to_utc(week,seconds,0)                
                #print (row[0] + " " + str(ts))
                _imu = imu0.copy()
                _imu['time'] = ts
                
                _imu['accZ'] = int(row[7])
                _imu['accY'] = -int(row[8])
                _imu['accX'] = int(row[9])
                _imu['gyrZ'] = int(row[10])
                _imu['gyrY'] = -int(row[11])
                _imu['gyrX'] = int (row[12].split('*')[0])
                imu.append(_imu)
                
                
                
                cnt[1] += 1
            elif ('#TIMEA' in row):
                point = point0.copy()
                # week seconds lat lon alt Nvel Evel Uvel Roll Pith Azim 
                week = int(row[5])
                seconds = float(row[6])
                ts = week_seconds_to_utc(week,seconds,0)                
                #print (ts)
                cnt[2] += 1
            elif ('#BESTGNSSPOSA' in row):
                point = point0.copy()
                # week seconds lat lon alt Nvel Evel Uvel Roll Pith Azim 
                week = int(row[5])
                seconds = float(row[6])
                ts = week_seconds_to_utc(week,seconds,0)                
                #print (ts)
                cnt[3] += 1

            elif ('#BESTPOSA' in row):
                point = point0.copy()
                # week seconds lat lon alt Nvel Evel Uvel Roll Pith Azim 
                week = int(row[5])
                seconds = float(row[6])
                ts = week_seconds_to_utc(week,seconds,0)                
                #print (ts)
                cnt[4] += 1
            elif ('#HEADING2A' in row):
                point = point0.copy()
                # week seconds lat lon alt Nvel Evel Uvel Roll Pith Azim 
                week = int(row[5])
                seconds = float(row[6])
                ts = week_seconds_to_utc(week,seconds,0)                
                #print (ts)
                cnt[5] += 1
            elif ('#RANGECMPA' in row):
                point = point0.copy()
                # week seconds lat lon alt Nvel Evel Uvel Roll Pith Azim 
                week = int(row[5])
                seconds = float(row[6])
                ts = week_seconds_to_utc(week,seconds,0)                
                #print (ts)
                cnt[6] += 1
                
            elif ('#RAWEPHEMA' in row):
                point = point0.copy()
                # week seconds lat lon alt Nvel Evel Uvel Roll Pith Azim 
                week = int(row[5])
                seconds = float(row[6])
                ts = week_seconds_to_utc(week,seconds,0)                
                #print (ts)
                cnt[7] += 1
            elif ('#RXCONFIGA' in row):
                week = 0                
                cnt[8] += 1
                
            else:
                print ('[[' + row[0] + ']]')
                row = next(pclreader)
                cnt[9] += 1

                
                
        
    return np.array(points), np.array(imu), cnt
#%%
def ouster_imu_reader(fname):
    imu0 = {'time':0, 'accX':0, 'accY':0, 'accZ':0,
            'gyrX':0, 'gyrY':0, 'gyrZ':0}
    imu = []           
    with open (fname) as csvfile:
        pclreader = csv.reader(csvfile, delimiter = ' ', skipinitialspace=True)
        for row in pclreader:
            _imu = imu0.copy()
            _imu['time'] = float(row[0])
            
            _imu['accZ'] = float(row[4])
            _imu['accY'] = float(row[3])
            _imu['accX'] = float(row[2])
            _imu['gyrZ'] = float(row[7])
            _imu['gyrY'] = float(row[6])
            _imu['gyrX'] = float (row[5])
            
            imu.append(_imu)
            
    return imu
    
#%%    
def nvs2lla(nvs):
    return np.array( [ np.array([n['lat'], n['lon'], n['alt_a'], n['quality']]) for n in nvs])
    

def write_pos (fname,array):
    with open(fname,'wt') as file:
    
        for pos in array:
            buf = '{},{},{},{}\n'.format(pos[0],pos[1],pos[2], pos[3]);\
            file.write(buf);
        file.close()
    
def lla2ecef(lla):
    ecef = [ navpy.lla2ecef(pos[0],pos[1],pos[2]) for pos in lla];
    return np.array(ecef);


def lla2ned(lla, ref=None):
    if (ref == None):
        ref = lla[0,:3];
    ned = np.array([ navpy.lla2ned(pos[0],pos[1],pos[2], ref[0], ref[1], ref[2]) for pos in lla])
    return ned;

def lla2enu(lla, ref=None):
    out = lla.copy()
    ned = lla2ned(lla, ref);
    enu = ned[:,[1,0,2]];
    enu [:,2] = 0 - enu[:,2];               
    out[:,:3]=enu
    return out
        
    
#%%

def nvs2lla_course(nvs):
    return np.array( [ np.array([n['lat'], n['lon'], n['alt_a'], 
                                 n['quality'], n['bl_course'], n['rmc_course'],
                                 n['imu_roll'], n['imu_pitch'], n['imu_yaw'] ]) 
                                for n in nvs])



def plot_courses(ax, enu_course, num, bImu = False, bBl = False, len = 100):
    x0 = enu_course[num][0]
    y0 = enu_course[num][1]

    #
    #rmc_course=np.radians(enu_course[num][5]) * -1
    bl_course=np.radians(enu_course[num][4]-90)  * -1
    imu_course=np.radians(enu_course[num][8]) * -1

    x1_ = 0
    y1_ = len
    #x1_rmc = x0 + x1_ * np.cos(rmc_course) - y1_ * np.sin(rmc_course)
    #y1_rmc = y0 + x1_ * np.sin(rmc_course) + y1_ * np.cos(rmc_course)
    #ax.plot([x0,x1_rmc], [y0,y1_rmc],color='r')

    
    if (bBl):
        x1_bl = x0 + x1_ * np.cos(bl_course) - y1_ * np.sin(bl_course)
        y1_bl = y0 + x1_ * np.sin(bl_course) + y1_ * np.cos(bl_course)
        ax.plot([x0,x1_bl], [y0,y1_bl],color='g')
    
    if (bImu):
        x1_imu = x0 + x1_ * np.cos(imu_course) - y1_ * np.sin(imu_course)
        y1_imu = y0 + x1_ * np.sin(imu_course) + y1_ * np.cos(imu_course)
        ax.plot([x0,x1_imu], [y0,y1_imu],color='r')
    
        