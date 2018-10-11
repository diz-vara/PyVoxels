# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 14:44:56 2018

@author: avarfolomeev
"""


pwrpak61b = read_ggaex_log('e:/Data/Voxels/201809_usa/test6_1a/IMU/test6_1.ggaex1')
o_imu61b = ouster_imu_reader('e:/data/Voxels/201809_usa/test6_1a/IMU/LidarTimestamps.txt')
time = np.array([ p['time'] for p in pwrpak61b])

time_i = time[22636]
time_o = o_imu61b[14874]['time']


time_i #1536336657.0880001
time_o #1536347409.745537
time_o - time_i # 10752.65753698349

time_diff_o_i = time_o - time_i;
len(pwrpak61b)
for i in range(len(pwrpak61b)):
    pwrpak61b[i]['time'] += time_diff_o_i;
    
write_IMUdata_from_NMEA(pwrpak61b, "test6_1b_corr.imu")
