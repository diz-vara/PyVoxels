# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 12:19:33 2018

@author: avarfolomeev
"""
import json

def build_dict(var):
    d = {
         "type_id": "opencv-matrix",
         "rows" : var.shape[0],
         "cols" : var.shape[1],
         "dt"   : "d",
         "data" : var.tolist()
         }
    return d     

def save_calib_json(fname, calib_dict):
  with open("cam2_calib.json","w") as jfile:
      
    json.dump({"cameraMatrix":build_dict(calib_dict["mtx"]), 
               "distCoeffs":build_dict(calib_dict["dist"]),
               "rot_vec":build_dict(calib_dict["rot"]),
               "t_vec":build_dict(calib_dict["t"])}, 
              jfile)  
    