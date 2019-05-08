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
         "data" : var.flatten().tolist()
         }
    return d     

def build_j_calib_dict(calib_dict):

    json_dict = {"cameraName":calib_dict["camera_name"],
            "cameraCenter":calib_dict["camera_center"],
            "cameraMatrix":build_dict(calib_dict["mtx"]), 
               "distCoeffs":build_dict(calib_dict["dist"]),
               "rot_vec":build_dict(calib_dict["rot"]),
               "t_vec":build_dict(calib_dict["t"])
               }
    if "cut_rows" in calib_dict.keys():
        json_dict["cut_rows"] = calib_dict["cut_rows"]
    return json_dict           

    
def save_calib_json(fname, front_calib_dict, back_calib_dict=None):
    
    if (back_calib_dict is None):
        calib_dict = build_j_calib_dict(front_calib_dict)
    else:
        calib_dict = {"Front":build_j_calib_dict(front_calib_dict),
                      "Back":build_j_calib_dict(back_calib_dict)}

    with open(fname,"w") as jfile:
        json.dump(calib_dict, jfile)  
    
        
def save_calib_list_json(fname, camera_list):
    
    calib_list = [build_j_calib_dict(camera) for camera in camera_list ]

    c_dict = {"numCameras":len(calib_list), "cameras":calib_list}
    with open(fname,"w") as jfile:
        json.dump(c_dict, jfile)  
            