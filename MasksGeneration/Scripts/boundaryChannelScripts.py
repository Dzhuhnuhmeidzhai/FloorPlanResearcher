import numpy as np
import cv2

#walls channel
def wchannel1(img):
  h_im, w_im = img.shape[0],img.shape[1]
  canvas = np.full((h_im,w_im),0)
  for i in range(h_im):
    for j in range(w_im):
      if  tuple(img[i][j]) == (255,255,255):
        canvas[i][j] = 1
  return canvas

#background channel (walls image)
def wchannel2(img):
  h_im, w_im = img.shape[0],img.shape[1]
  canvas = np.full((h_im,w_im),0)
  for i in range(h_im):
    for j in range(w_im):
      if  tuple(img[i][j]) == (0,0,0):
        canvas[i][j] = 1
  return canvas

#doorsWindows channel
def dwchannel1(img):
  h_im, w_im = img.shape[0],img.shape[1]
  canvas = np.full((h_im,w_im),0)
  for i in range(h_im):
    for j in range(w_im):
      if  tuple(img[i][j]) == (255,255,255):
        canvas[i][j] = 1
  return canvas

#background channel (doorsWindows image)
def dwchannel2(img):
  h_im, w_im = img.shape[0],img.shape[1]
  canvas = np.full((h_im,w_im),0)
  for i in range(h_im):
    for j in range(w_im):
      if  tuple(img[i][j]) == (0,0,0):
        canvas[i][j] = 1
  return canvas

#resize individual channel
#special case when background channel
def preprocess_wallsdw_mask_channel(img, ischannel2=False):
  img = img.astype('float32')
  #create empty canvas
  h_out = w_out = 512
  canvas = np.full((h_out,w_out),0)
  if ischannel2 == True:
    canvas = np.full((h_out,w_out),1)
  h_im, w_im = img.shape[0],img.shape[1]
  #find resized image height, image width
  newh = int(h_im * min(h_out/h_im, w_out/w_im))
  neww = int(w_im * min(h_out/h_im, w_out/w_im))
  #layout on canvas
  img_rs = cv2.resize(img,(neww,newh),interpolation=cv2.INTER_LINEAR)
  canvas[(h_out-newh)//2:(h_out-newh)//2 + newh, (w_out-neww)//2:(w_out-neww)//2 + neww] = img_rs
  return canvas