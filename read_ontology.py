# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:18:12 2019

@author: avarfolomeev
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import struct

from collections import namedtuple
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




def read_ontology(fname, delimiter = ','):

    with open (fname) as csvfile:
        ont_reader = csv.reader(csvfile, delimiter = delimiter)
        labels = [ sLabel(  'unlabeled'      ,  0 , (    0,    0,    0) )]
        next(ont_reader,None)
        for row in ont_reader:
            hex_color = row[0]
            color = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
            name = row[2]
            code = int(row[3])
            labels.append(sLabel(name, code, color))
        
    colors = np.array([label.color for label in labels]).astype(np.uint8)
    
    return labels, colors
    
def read_ontology_vlg(fname, delimiter = ','):

    with open (fname) as csvfile:
        ont_reader = csv.reader(csvfile, delimiter = delimiter)
        labels = [ sLabel(  'unlabeled'      ,  0 , (    0,    0,    0) )]
        next(ont_reader,None)
        for row in ont_reader:
            hex_color = row[1]
            color = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
            name = row[2]
            code = int(row[0])
            labels.append(sLabel(name, code, color))
        
    colors = np.array([label.color for label in labels]).astype(np.uint8)
    
    return labels, colors
    
    
