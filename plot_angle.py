# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 22:52:57 2018

@author: diz
"""
from numpy import sin,cos

global a2
def plot_angle(num, c='r'):
    x0 = enu[num,0];
    y0 = enu[num,1];
    x1 = x0 + 50 * cos(y[num]);
    y1 = y0 + 50 * sin(y[num] * -1);

    a2.scatter(x0,y0,edgecolors='face',marker='o',c=c,s=20)
    a2.plot([x0,x1],[y0,y1],c=c)
    