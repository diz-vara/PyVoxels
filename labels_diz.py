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

labels_diz = [
    #       name                id    color
    sLabel(  'unlabeled'      ,  0 , (    0,    0,    0) ),
    sLabel(  'sky'            ,  1 , ( 0x46, 0x82, 0xB4) ),
    sLabel(  'road'           ,  2 , ( 0x80, 0x40, 0x80) ),
    sLabel(  'sidewalk'       ,  3 , ( 0xF4, 0x23, 0xE8) ),
    # xu 4 -> 3
    sLabel(  'Lane markers'   ,  5 , ( 0xff, 0xFF, 0x8C) ), #4
    sLabel(  'Railway'        ,  6 , ( 0x37, 0x37, 0x37) ), #5
    sLabel(  'Grass'          ,  7 , ( 0x98, 0xFB, 0x98) ), #6
    # xu 8 -> 9 sLabel(  'Tree'           ,  8 , (107,142, 35) ),
    sLabel(  'Vegetation'     ,  9 , ( 0x32, 0x8E, 0x32) ), #7
    sLabel(  'Building'       , 10 , ( 0x46, 0x46, 0x46) ), #8
    sLabel(  'Bridge'         , 11 , ( 0x96, 0x64, 0x64) ), #9
    sLabel(  'Construction'   , 12 , ( 0x99, 0x99, 0x99) ), #10
    sLabel(  'Panel'          , 13 , ( 0xC8, 0xC8,    0) ), #11
    sLabel(  'traffic light'  , 14 , ( 0xFA, 0xAA, 0x1E) ), #12
    # xu 15 -> 12 sLabel(  'Fence'          , 15 , (210,153,153) ),
    # xu 16 -> 12 sLabel(  'Construction'   , 16 , (180,185,180) ),
    sLabel(  'Transport'      , 17 , (    0,    0, 0x8E) ), #13
    # xu 18-> 17 sLabel(  'Truck'          , 18 , (  0,  0, 70) ),
    #xu 19 -> 17 sLabel(  'Bus'            , 19 , (  0, 60,100) ),
    #xu 20 -> 17 sLabel(  'Train'          , 20 , (  0, 80,100) ),
    sLabel(  'Pedestrian'     , 21 , (0xDC, 0x14, 0x3C) ), #14
    #xu 22 -> 21 sLabel(  'Children'       , 22 , (220, 20, 60) ),
    sLabel(  'Rider'          , 23 , (0xFF,    0,    0) ), #15
    sLabel(  'bicycle'        , 24 , (0x77, 0x0B, 0x20) ), #16
    #xu 25 -> 23 sLabel(  'Motoryclist'    , 25 , (255,  0,  0) ),
    #xu 26 -> 24 sLabel(  'Motorcycle'     , 26 , (  0,  0,230) ),
    sLabel(  'Animal'         , 27 , (0xFA, 0x0F, 0x32) ), #17
    sLabel(  'Movable'        , 28 , (0x96, 0x32,    0) )  #18
]

colors_diz = np.array([label.color for label in labels_diz]).astype(np.uint8)

#%%
def reCode (base_dir, in_dir, out_dir, table):
    in_dir = os.path.join(base_dir, in_dir)
    out_dir = os.path.join(base_dir, out_dir)

    try:
        os.makedirs(out_dir)
    except:
        pass


    print('Loading masks from ' + in_dir)

    
    

    im_files = sorted(os.listdir(in_dir))
    cnt = 0
    end_string = ' from ' + str(len(im_files))
    
    
    
    for label_file in im_files:
        print(str(cnt) + end_string)
        cnt = cnt+1
        label_in = cv2.imread(os.path.join(in_dir, label_file),-1)
        if (len(label_in.shape) > 2):   #from multi-channel labels (synthia)
            label_in = label_in[:,:,-1] #take only last channel
        label_in[label_in >= len(table) ] = 0
        label_out = table[label_in]
        cv2.imwrite(os.path.join(out_dir, label_file), label_out)

#%%


def xu2diz(base_dir, in_dir, out_dir):
    _xu2diz = np.array([0, 1, 2, 3, 3, 4, 5, 6, 7, 7, 8 ,9, 10, 11, 12, 10, 10, 
          13, 13, 13, 13, 14, 14, 15, 16, 15, 16, 17, 18]).astype(np.uint8)
    
    reCode(base_dir, in_dir,out_dir, _xu2diz);

    
    
#%%
def synthia2diz(base_dir, in_dir, out_dir):
    syn2diz = np.array([
                        10, #void -> construction
                        1, #sky
                        8, #building
                        2, #road
                        3, #sidewalk
                        10, #Fence -> construction
                        7, #vegetation
                        10, #pole -> construction
                        13, #car->transport
                        11, #sign
                        14, #pedestrian
                        16, #bicycle
                        4, #lanemarking
                        0,
                        0,
                        12 #traffic light
                        ]).astype(np.uint8)
    
    reCode(base_dir, in_dir,out_dir, syn2diz);

#%%
def colors2label(base_dir, in_dir, out_dir, colors):
    
    in_dir = os.path.join(base_dir, in_dir)
    out_dir = os.path.join(base_dir, out_dir)

    try:
        os.makedirs(out_dir)
    except:
        pass


    print('Loading masks from ' + in_dir)

    
    

    im_files = sorted(os.listdir(in_dir))
    cnt = 0
    end_string = ' from ' + str(len(im_files))
    
    
    
    
    
    for label_file in im_files:
        print(str(cnt) + end_string)
        cnt = cnt+1
        colors_in = cv2.imread(os.path.join(in_dir, label_file))
        colors_in = cv2.cvtColor(colors_in,cv2.COLOR_RGB2BGR)
        
        
        
        sh = colors_in.shape

        label = np.zeros ((sh[0], sh[1]), dtype = np.uint8)

        
        for idx in range(len(colors)):
            color = colors[idx]
            s = (colors_in == color).all(axis=2)
            label[s] = idx            
        
        cv2.imwrite(os.path.join(out_dir, label_file), label)
          
#%%
def diz2cityScapes(base_dir, in_dir, out_dir):
    _diz2cs = np.array([
                       0,   #  0 unmarked 
                       23,  #  1 sky
                       7,   #  2 road
                       8,   #  3 sidewalk
                       65,  #  4 lane marker
                       10,  #  5 railway
                       22,  #  6 terrain
                       21,  #  7 vegetaion
                       11,  #  8 building
                       15,  #  9 --bridge
                       14,  # 10 construction (guard rail)
                       20,  # 11 sign
                       19,  # 12 traffick light
                       26,  # 13 transport (car)
                       24,  # 14 pedestrian
                       25,  # 15 rider
                       33,  # 16 bicycle
                        5,  # 17 animal (dynamic)
                        4   # 18 static
                       ]).astype(np.uint8)

    in_dir = os.path.join(base_dir, in_dir)
    out_dir = os.path.join(base_dir, out_dir)

    try:
        os.makedirs(out_dir)
    except:
        pass


    print('Loading "diz" labels from ' + in_dir)
    
    
    im_files = sorted(os.listdir(in_dir))
    cnt = 0
    end_string = ' from ' + str(len(im_files))
    

    for label_file in im_files:
        print(str(cnt) + end_string)
        cnt = cnt+1
        label_in = cv2.imread(os.path.join(in_dir, label_file),-1)

        label_out = _diz2cs[label_in]
        cv2.imwrite(os.path.join(out_dir, label_file), label_out)


 #%%
