import tensorflow
import numpy as np


def saveBoundaryMasks(bmasks, mask_save_path, im_save_path, train):
	#bmasks: list of boundary masks (each element of list of shape (512,512,3))
	#mask_save_path: path where you want to store array with all target boundary masks in it
	#im_save_path: path where you want to store each boundary mask as image
	#train: generating for training data if "True", generating for testing data if "False"
	##############################
	#create boundary masks list
	bma = []
	print("Serial no. of boundaries image being saved:")
	for i in range(len(bmasks)):
		print(i)
		#save image form of boundaries mask
		im_i = tensorflow.keras.preprocessing.image.array_to_img(bmasks[i])
		im_i_path = im_save_path + str(i) + ".jpg"
		tensorflow.keras.preprocessing.image.save_img(im_i_path,im_i)
		#save masks form
		targ_i = tensorflow.reshape(bmasks[i],(1,512,512,3))
		bma.append(targ_i.numpy())
	#save input image and boundary masks
	if train == True:
		train_bmasks = np.asarray(bma)
		np.save(mask_save_path+'train_bmasks.npy',train_bmasks)
	else:
		train_bmasks = np.asarray(bma)
		np.save(mask_save_path+'test_bmasks.npy',train_bmasks)
