# -*- coding: utf-8 -*-
"""
Created on Mon May 21 09:36:50 2018

@author: avarfolomeev
"""

with open('20180525_ride2a1.c.csv','wt') as file:
    
    for pos in posRD2:
        buf = '{},{},{}\n'.format(pos[0],pos[1],pos[2]);\
        file.write(buf);
    file.close()
    
    