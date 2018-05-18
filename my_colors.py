# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:07:04 2018

@author: avarfolomeev
"""



def getcolor(n):
    
    cmap = ['blue','dodgerblue','cyan', 'lightgreen', 'green', 'yellow', 
            'orange', 'red', 'magenta', 'brown',  'darkgray'];
    n = n%len(cmap)
    return cmap[n]
        
        
        
        