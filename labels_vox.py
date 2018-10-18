#!/usr/bin/python
#
# Cityscapes labels
#

from collections import namedtuple
import cv2
import numpy as np

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
sLabel = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------
#%%
# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels_vox = [
    #       name                id    color
    sLabel(  'unlabeled'        ,  0 , (    0,    0,    0) ),
    sLabel(  'sky'              ,  1 , ( 0x46, 0x82, 0xB4) ),
    sLabel(  'Vegetation'       ,  2 , ( 0x32, 0x8E, 0x32) ), 
    sLabel(  'Terrain, grass'   ,  3 , ( 0x98, 0xFB, 0x98) ), 
    sLabel(  'Building,constr.' ,  4 , ( 0x46, 0x46, 0x46) ), 
    sLabel(  'Road'             ,  5 , ( 0x80, 0x40, 0x80) ),
    sLabel(  'Sidewalk'         ,  6 , ( 0xF4, 0x23, 0xE8) ),
    sLabel(  'Road marking'     ,  7 , ( 0xFF, 0xFF, 0x8C) ), 
    sLabel(  'Zebra Crosswalk'  ,  8 , ( 0x96, 0xB5, 0xCA) ),
    sLabel(  'Manhole, hatch'   ,  9 , ( 0x59, 0x53, 0x92) ),
    sLabel(  'Speed bump'       , 10 , ( 0x9B, 0xDD, 0x2E) ),
    sLabel(  'Railway'          , 11 , ( 0xCD, 0xD2, 0x9C) ),
    sLabel(  'Pole'             , 12 , ( 0x2B, 0x9B, 0xCD) ),
    sLabel(  'Pole Arm'         , 13 , ( 0xBD, 0x78, 0x65) ),
    sLabel(  'Street Lamp'      , 14 , ( 0xFF, 0x55, 0x00) ),
    sLabel(  'Fence'            , 15 , ( 0xF3, 0xD2, 0x34) ),
    sLabel(  'Guard rail'       , 16 , ( 0xD4, 0xA8, 0x9E) ),
    sLabel(  'Traffic light'    , 17 , ( 0xAC, 0x2C, 0xCD) ),
    sLabel(  'Traffic sign'     , 18 , ( 0xC8, 0xC8,    0) ), 
    sLabel(  'Movable furniture', 19 , ( 0xFA, 0xA9, 0x1E) ), 
    sLabel(  'Object'           , 20 , ( 0x99, 0x99, 0x99) ),
    sLabel(  'Transport'        , 21 , (    0,    0, 0x8E) ), 
    sLabel(  'Bicycle,motorbike', 22 , ( 0x77, 0x0B, 0x20) ),
    sLabel(  'Person'           , 23 , ( 0xDC, 0x14, 0x3C) ),
    sLabel(  'Rider'            , 24 , ( 0xFF,    0,    0) ), 
    sLabel(  'Animal'           , 25 , ( 0xCA, 0x96, 0xC7) ), 
    sLabel(  'Curb'             , 26 , ( 0x6c, 0xBA, 0xC1) ), 
    sLabel(  'Drop Curb'        , 27 , ( 0x74, 0x6C, 0xC1) )

]

colors_vox = np.array([label.color for label in labels_vox]).astype(np.uint8)

