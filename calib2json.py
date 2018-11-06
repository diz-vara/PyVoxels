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

def build_j_calib_dict(calib_dict):

    return {"cameraMatrix":build_dict(calib_dict["mtx"]), 
               "distCoeffs":build_dict(calib_dict["dist"]),
               "rot_vec":build_dict(calib_dict["rot"]),
               "t_vec":build_dict(calib_dict["t"])}

    
def save_calib_json(fname, front_calib_dict, back_calib_dict):
    
    calib_dict = {"Front":build_j_calib_dict(front_calib_dict),
                  "Back":build_j_calib_dict(back_calib_dict)}

    with open(fname,"w") as jfile:
        json.dump(calib_dict, jfile)  
    