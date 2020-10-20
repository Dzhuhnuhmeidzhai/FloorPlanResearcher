import numpy as np
import tensorflow 
import boundaryChannelScripts
from boundaryChannelScripts import *


#get boundary masks i.e. walls, doors, windows
def getBoundaryMasks(walls_images, doorswindows_images):
    #create placeholder for boundary masks
    boundary_masks_list = []
    print("Serial no. of boundaries mask being processed: ")
    #create a boundary masks for every (walls, doorsWindows) image pair
    for i in range(len(walls_images)):
        print(i)
        #get walls image wall and background channels
        wch1 = wchannel1(walls_images[i])
        wch2 = wchannel2(walls_images[i])
        #get doorsWindows image doorsWindows and background channel
        dwch1 = dwchannel1(doorswindows_images[i])
        dwch2 = dwchannel2(doorswindows_images[i])
        #resize each channel individually
        res_wc1 = preprocess_wallsdw_mask_channel(wch1)
        res_wc2 = preprocess_wallsdw_mask_channel(wch2,True)
        res_dwc1 = preprocess_wallsdw_mask_channel(dwch1)
        res_dwc2 = preprocess_wallsdw_mask_channel(dwch2,True)
        #stack background of walls image, walls of walls image, doorswindows of doorswindows image
        bdary_im_arr = np.stack((res_dwc2, res_dwc1, res_wc1), axis = -1)
        #resizing correction
        for j in range(512):
            for k in range(512):
                #if nothing, make wall (fill gaps)
                if tuple(bdary_im_arr[j][k]) == (0,0,0):
                    bdary_im_arr[j][k][0] = 0
                    bdary_im_arr[j][k][1] = 0
                    bdary_im_arr[j][k][2] = 1
                #if both background and doorsWindows, make doorsWindows
                if tuple(bdary_im_arr[j][k]) == (1,1,0):
                    bdary_im_arr[j][k][0] = 0
                    bdary_im_arr[j][k][1] = 1
                    bdary_im_arr[j][k][2] = 0
                #if both background and wall, make wall
                if tuple(bdary_im_arr[j][k]) == (1,0,1):
                    bdary_im_arr[j][k][0] = 0
                    bdary_im_arr[j][k][1] = 0
                    bdary_im_arr[j][k][2] = 1
        boundary_masks_list.append(bdary_im_arr)
    return boundary_masks_list