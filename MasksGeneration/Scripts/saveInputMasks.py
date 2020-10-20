import numpy as np
import tensorflow


def saveInputMasks(imasks, mask_save_path, im_save_path, train):
  #imasks: list of input masks (each element of list of shape (512,512,3))
  #mask_save_path: path where you want to store array with all input masks in it
  #im_save_path: path where you want to store each input mask as image
  #train: generating for training data if "True", generating for testing data if "False"
  ##############################
  #create input placeholder list
  inp = []
  print("Serial no. of input image being saved:")
  for i in range(len(imasks)):
    print(i)
    #save image form of input mask
    im_i = tensorflow.keras.preprocessing.image.array_to_img(imasks[i])
    im_i_path = im_save_path + str(i) + ".jpg"
    tensorflow.keras.preprocessing.image.save_img(im_i_path,im_i)
    #save array form
    input_i = tensorflow.reshape(imasks[i],(1,512,512,3))
    inp.append(input_i.numpy())
  #save input image array
  if train == True:
    train_input = np.asarray(inp)
    np.save(mask_save_path+'train_inputs.npy',train_input)
  else:
    train_input = np.asarray(inp)
    np.save(mask_save_path+'test_inputs.npy',train_input)

