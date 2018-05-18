# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:56:28 2018

@author: avarfolomeev
"""

for k in range(12):
    pt1,neds,diffs=translate_points(points[frames[k+1]:frames[k+2]], 
                                          rm, IMUdata21)
    a2.scatter(pt1[:,0], pt1[:,1],marker='.',
                  edgecolors='face',color=getcolor(k),s=5)
    ax_3d.scatter(pt1[:,0], pt1[:,1], pt1[:,2],
                  marker='.',edgecolors='face',color=getcolor(k),s=3)