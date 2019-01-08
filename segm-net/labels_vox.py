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
    sLabel(  'Barrier',           19 , ( 0xFA, 0xA9, 0x1E) ), 
    sLabel(  'Object'           , 20 , ( 0x99, 0x99, 0x99) ),
    sLabel(  'Transport'        , 21 , (    0,    0, 0x8E) ), 
    sLabel(  'Bicycle,motorbike', 22 , ( 0x77, 0x0B, 0x20) ),
    sLabel(  'Person'           , 23 , ( 0xDC, 0x14, 0x3C) ),
    sLabel(  'Animal'           , 25 , ( 0xCA, 0x96, 0xC7) ), 
    sLabel(  'Curb'             , 26 , ( 0x6c, 0xBA, 0xC1) ), 
    sLabel(  'Drop Curb'        , 27 , ( 0x74, 0x6C, 0xC1) ),
    sLabel(  'Portable'         , 28 , ( 0xC0, 0x30, 0x00) )
  

]

colors_vox = np.array([label.color for label in labels_vox]).astype(np.uint8)

ids = [l.id for l in labels_vox]
max_id = max(ids)

indexes = np.zeros(max_id + 1,dtype=int)

for i in range(max_id+1):
    if (i in ids):
        indexes[i] = ids.index(i)
    


#%%
def idx2label(labels,idx):
    if (idx >= len(labels)):
        idx = 0
    return labels[idx].id 
    
def idx2label_vox(idx):
    return idx2label(labels_vox, idx)


def label_vox2idx(label):
    if (label >= len(indexes)):
        label = 0;
    return indexes[label]   
    

def create_dirs(base_dir, in_dir, out_dir):
    in_dir = os.path.join(base_dir, in_dir)
    out_dir = os.path.join(base_dir, out_dir)
    try:
        os.makedirs(out_dir)
    except:
        pass
    return in_dir, out_dir

def voxelLabels2idx(base_dir, in_dir, out_dir):
    in_dir, out_dir = create_dirs(base_dir, in_dir, out_dir);
    print('Loading "voxelmaps" labels from ' + in_dir)

    maxlabel = max(ids)
    im_files = sorted(os.listdir(in_dir))
    cnt = 0
    end_string = ' from ' + str(len(im_files))
    

    for label_file in im_files:
        print(str(cnt) + end_string)
        cnt = cnt+1
        label_in = cv2.imread(os.path.join(in_dir, label_file),-1)
        if (len(label_in.shape) > 2):  #if RGB, use only R
            label_in = label_in[:,:,2]
        label_in[label_in > maxlabel] =  0
        
        label_out = indexes[label_in]
        cv2.imwrite(os.path.join(out_dir, label_file), label_out)
        
#%%
def voxelLabels2diz(base_dir, in_dir, out_dir):
    _vox2diz = np.array([
                     0, #unlabled
                     1, #sky
                     7, #vegetation
                     6, #grass
                     8, #building
                     2, #road
                     3, #sidewalk
                     4, #road marking
                     4, #crosswalk -> road marking
                     10, #manhole -> object
                     10, # speed bump -> object
                     5,  #railway
                     10, #pole->object
                     10, #pole arm -> object
                     10, #street lamp-> object
                     10, #fence -> object
                     10, #guard rail->object
                     12, #traffic light
                     11, #traffic sign
                     10, #road furniture -> object
                     10, #object
                     13, #transport
                     16, #bicycle
                     14, #person
                     14, #rider -> person
                     17, #animal
                     10, #curb -> object
                     10,  #drop curb -> object
                     18
                     ]).astype(np.uint8)

    in_dir = os.path.join(base_dir, in_dir)
    out_dir = os.path.join(base_dir, out_dir)
    ncolors = len(_vox2diz)

    try:
        os.makedirs(out_dir)
    except:
        pass


    print('Loading "voxelmaps" labels from ' + in_dir)
    
    
    im_files = sorted(os.listdir(in_dir))
    cnt = 0
    end_string = ' from ' + str(len(im_files))
    

    for label_file in im_files:
        print(str(cnt) + end_string)
        cnt = cnt+1
        label_in = cv2.imread(os.path.join(in_dir, label_file),-1)
        if (len(label_in.shape) > 2):  #if RGB, use only R
            label_in = label_in[:,:,2]
        label_in[label_in >= ncolors] =  0
        label_out = _vox2diz[label_in]
        cv2.imwrite(os.path.join(out_dir, label_file), label_out)
                    
#%%
weights = np.array([  0.00000000e+00,   2.33448024e-05,   1.68016065e-05,
         5.25011659e-05,   1.53376239e-05,   6.59329070e-06,
         9.61631283e-05,   1.37839701e-04,   1.10753576e-04,
         2.41516528e-03,   6.85293583e-01,   6.74822204e-02,
         3.24308120e-04,   2.67684523e-03,   1.24839988e-02,
         9.51945986e-05,   3.49990642e-04,   1.09327204e-03,
         4.00805304e-04,   1.50683095e-04,   3.33722665e-04,
         3.40661715e-05,   2.25408591e-03,   5.60762072e-04,
         0.00000000e+00,   2.18260616e-01,   2.18282451e-04,
         3.32657758e-03,   1.78648527e-03])

weights_idx = np.array([0.00000000e+00,   2.33448024e-05,   1.68016065e-05,
         5.25011659e-05,   1.53376239e-05,   6.59329070e-06,
         9.61631283e-05,   1.37839701e-04,   1.10753576e-04,
         2.41516528e-03,   6.85293583e-01,   6.74822204e-02,
         3.24308120e-04,   2.67684523e-03,   1.24839988e-02,
         9.51945986e-05,   3.49990642e-04,   1.09327204e-03,
         4.00805304e-04,   1.50683095e-04,   3.33722665e-04,
         3.40661715e-05,   2.25408591e-03,   5.60762072e-04,
         2.18260616e-01,   2.18282451e-04,   3.32657758e-03,
         1.78648527e-03]) 

       