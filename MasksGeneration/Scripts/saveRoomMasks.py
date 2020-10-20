import tensorflow
import numpy as np
import roomMaskToImage
from roomMaskToImage import *


def saveRoomMasks(rmasks, mask_save_path, im_save_path, train):
	#rmasks: list of room masks (each element of list of shape (512,512,8))
	#mask_save_path: path where you want to store array with all target room masks in it
	#im_save_path: path where you want to store each room mask as image
	#train: generating for training data if "True", generating for testing data if "False"
	##############################
	#create room masks list
	rma = []
	print("Serial no. of room mask being saved: ")
	for i in range(len(rmasks)):
		print(i)
		#get image
		room_ar_i = mask2visible(rmasks[i])
		room_im_i = tensorflow.keras.preprocessing.image.array_to_img(room_ar_i)
		room_im_i_path = im_save_path + str(i) + ".jpg"
		tensorflow.keras.preprocessing.image.save_img(room_im_i_path,room_im_i)
		#get mask
		rmask_i = tensorflow.reshape(rmasks[i],(1,512,512,8))
		rma.append(rmask_i.numpy())
  	#save room masks
	if train == True:
		train_rmasks = np.asarray(rma)
		np.save(mask_save_path+'train_rmasks.npy',train_rmasks)
	else:
		train_rmasks = np.asarray(rma)
		np.save(mask_save_path+'test_rmasks.npy',train_rmasks)