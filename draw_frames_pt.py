# -*- coding: utf-8 -*-
"""
Created on Tue May 15 12:42:11 2018

@author: avarfolomeev
"""

cmap = ['blue','dodgerblue','cyan', 'lightgreen', 'green', 'yellow', 
        'orange', 'red', 'magenta', 'brown',  'darkgray'];
        
        

ax1.cla()

cnt = 0
rm = extract_quaternion(imuR1[0:2])[0].rotation_matrix
for frame in range(1,4):
    pt_ = [[p['X'], p['Y'], p['Z']] for p in pts if p['frame']==frame]
    ptr = np.array(pt_ * imu_to_velo );

    filter_a = build_filter(ptr, limits)
    ax1.scatter(ptr[filter_a,0], 
               ptr[filter_a,1],
               marker='.',
               edgecolors='face',
               color=cmap[frame],s=1)
