# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:51:43 2019

@author: avarfolomeev
"""

min_loss = 1

try:
    min_loss = float(open('min_lossd.txt').read())
except:
    0;

print (min_loss)    
    
    