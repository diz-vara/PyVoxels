# -*- coding: utf-8 -*-
"""
Created on Wed May 16 22:37:35 2018

https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
"""
import numpy as np
from pyquaternion import Quaternion


def toQuaternion(yaw, pitch, roll):
	cy = np.cos(yaw * 0.5);
	sy = np.sin(yaw * 0.5);
	cr = np.cos(roll * 0.5);
	sr = np.sin(roll * 0.5);
	cp = np.cos(pitch * 0.5);
	sp = np.sin(pitch * 0.5);

	w = cy * cr * cp + sy * sr * sp;
	x = cy * sr * cp - sy * cr * sp;
	y = cy * cr * sp + sy * sr * cp;
	z = sy * cr * cp - cy * sr * sp;
	return Quaternion(w,x,y,z);


def toQuaternion(rpy):
	cy = np.cos(rpy[2] * 0.5);
	sy = np.sin(rpy[2] * 0.5);
	cr = np.cos(rpy[0] * 0.5);
	sr = np.sin(rpy[0] * 0.5);
	cp = np.cos(rpy[1] * 0.5);
	sp = np.sin(rpy[1] * 0.5);

	w = cy * cr * cp + sy * sr * sp;
	x = cy * sr * cp - sy * cr * sp;
	y = cy * cr * sp + sy * sr * cp;
	z = sy * cr * cp - cy * sr * sp;
	return Quaternion(w,x,y,z);

#       __QUATEULERFUNCTIONS__
#       This file implements functions for handling 
#       and manipulating quaternios and Euler Angles
#
#       Authors: 
#       Kostas Alexis (kalexis@unr.edu)
# http://www.kostasalexis.com/frame-rotations-and-representations.html

from numpy import *
import numpy as np

def quat2r(q_AB=None):
    C_AB = np.zeros((3,3))
    C_AB[0,0] = q_AB[0]*q_AB[0] - q_AB[1]*q_AB[1] - q_AB[2]*q_AB[2] + q_AB[3]*q_AB[3]
    C_AB[0,1] = q_AB[0]*q_AB[1]*2.0 + q_AB[2]*q_AB[3]*2.0
    C_AB[0,2] = q_AB[0]*q_AB[2]*2.0 - q_AB[1]*q_AB[3]*2.0
    
    C_AB[1,0] = q_AB[0]*q_AB[1]*2.0 - q_AB[2]*q_AB[3]*2.0
    C_AB[1,1] = -q_AB[0]*q_AB[0] + q_AB[1]*q_AB[1] - q_AB[2]*q_AB[2] + q_AB[3]*q_AB[3]
    C_AB[1,2] = q_AB[0]*q_AB[3]*2.0 - q_AB[1]*q_AB[2]*2.0
    
    C_AB[2,0] = q_AB[0]*q_AB[2]*2.0 + q_AB[1]*q_AB[3]*2.0
    C_AB[2,1] = q_AB[0]*q_AB[3]*(-2.0) + q_AB[1]*q_AB[2]*2.0
    C_AB[2,2] = -q_AB[0]*q_AB[0] - q_AB[1]*q_AB[1] + q_AB[2]*q_AB[2] + q_AB[3]*q_AB[3]
    return C_AB

def quat2rpy(q_AB=None):
    C = quat2r(q_AB)
    theta = np.arcsin(-C[2,0])
    phi = np.arctan2(C[2,1],C[2,2])
    psi = np.arctan2(C[1,0],C[0,0])
    rpy = np.zeros((3,1))
    rpy[0] = phi
    rpy[1] = theta
    rpy[2] = psi
    return rpy

def rpy2quat(rpy=None):
    r = rpy[0] 
    p = rpy[1]
    y = rpy[2]
    cRh = np.cos(r/2)
    sRh = np.sin(r/2)
    cPh = np.cos(p/2)
    sPh = np.sin(p/2)
    cYh = np.cos(y/2)
    sYh = np.sin(y/2)
    qs_cmpl = np.array([ -(np.multiply(np.multiply(sRh,cPh),cYh) - np.multiply(np.multiply(cRh,sPh),sYh)),
                         -(np.multiply(np.multiply(cRh,sPh),cYh) + np.multiply(np.multiply(sRh,cPh),sYh)),
                         -(np.multiply(np.multiply(cRh,cPh),sYh) - np.multiply(np.multiply(sRh,sPh),cYh)),
                         np.multiply(np.multiply(cRh,cPh),cYh) + np.multiply(np.multiply(sRh,sPh),sYh)])
    qs = np.real(qs_cmpl)
    return qs

def r2rpy(C=None):
    theta = np.arcsin(-C[2,0])
    phi = np.arctan2(C[2,1],C[2,2])
    psi = np.arctan2(C[1,0],C[0,0])
    rpy = np.zeros((3,1))
    rpy[0] = phi
    rpy[1] = theta
    rpy[2] = psi
    return rpy

def normalized(x=None):
    y=x/np.sqrt(np.dot(x,x))
    return y