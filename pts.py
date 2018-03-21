# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 16:14:15 2018

@author: avarfolomeev
"""
pts = []

for i in range(10):
    for j in range(10):
        pts.append([X[i,j]+avg[0],Y[i,i]+avg[1],Z[i,j]+avg[2]])
        
pts = np.array(pts)
        