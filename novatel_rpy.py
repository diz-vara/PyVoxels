# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:56:12 2018

@author: avarfolomeev
"""

def novatel_rpy2matirx(rpy):
    rpy  = np.radians(rpy)
    r = rpy[0]
    p = rpy[1]
    y = rpy[2]
    
    cy = np.cos(y)
    sy = np.sin(y)
    
    cp = np.cos(p)
    sp = np.sin(p)

    cr = np.cos(r)
    sr = np.sin(r)
    
    
    res = np.matrix( [ [cy*cr - sy*sp*sr, -sy*cp, cy*sr + sy*sp*cr],
                       [sy*cr + cy*sp*sr,  cy*cp, sy*sr - cy*sp*cr],
                       [  -cp * sr,        sp,          cp*cr]
                      ])
    
    return res
    
    