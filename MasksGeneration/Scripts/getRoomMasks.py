import tensorflow
import numpy as np
import roomChannelScripts
from roomChannelScripts import *


#get room-type mask
def getRoomMasks(rooms_images):
    #create a placeholder for room masks
    room_masks_list = []
    #create room mask from each room image
    print("Serial no. of room mask being processed: ")
    for i in range(len(rooms_images)):
        print(i)
        #get individual channel
        rc1 = rchannel1(rooms_images[i])
        rc2 = rchannel2(rooms_images[i])
        rc3 = rchannel3(rooms_images[i])
        rc4 = rchannel4(rooms_images[i])
        rc5 = rchannel5(rooms_images[i])
        rc6 = rchannel6(rooms_images[i])
        rc7 = rchannel7(rooms_images[i])
        rc8 = rchannel8(rooms_images[i])
        #resize individual channel
        res_rc1 = preprocess_room_mask_channel(rc1)
        res_rc2 = preprocess_room_mask_channel(rc2)
        res_rc3 = preprocess_room_mask_channel(rc3)
        res_rc4 = preprocess_room_mask_channel(rc4)
        res_rc5 = preprocess_room_mask_channel(rc5)
        res_rc6 = preprocess_room_mask_channel(rc6)
        res_rc7 = preprocess_room_mask_channel(rc7)
        res_rc8 = preprocess_room_mask_channel(rc8,True)
        #create room mask
        room_mask_arr = np.stack((res_rc1, res_rc2, res_rc3, res_rc4, res_rc5, res_rc6, res_rc7,res_rc8), axis = -1)
        room_masks_list.append(room_mask_arr)
    return room_masks_list