# -*- coding: utf-8 -*-
"""
Created on Tue May 15 14:57:22 2018

@author: avarfolomeev
"""

import os.path
import struct


def save_points(path, points, frames,colors=None):
    format_4f = 'fffBBBB'
    
    frame_path = os.path.join(path,'frames');
    if os.path.exists(frame_path) is False:
        os.makedirs(frame_path)

    point_color = np.zeros(4,np.uint8);    
    if (colors is not None):
        ext = '.velorgbTcs';
    else:
        ext = '.bin'

    name = 'points_out' + ext;    
    one_file = open(os.path.join(path,name),'wb')
    for f in range (frames[-1]+1):
        pts = points[frames==f,:]
        if (colors is not None):
            fcolors = colors[frames==f,:]

        fname = os.path.join(frame_path,'{:08d}'.format(f))
        file = open(fname+ext,'wb')
        for pt_idx in range(len(pts)):
            pt = pts[pt_idx];
            if (colors is not None):
                point_color = fcolors[pt_idx,:];
            bf = struct.pack(format_4f,pt[0], pt[1], pt[2], 
                             point_color[0],point_color[1],
                             point_color[2],point_color[3]);
            file.write(bf);
            one_file.write(bf);
        file.close();
    one_file.close();

                         

