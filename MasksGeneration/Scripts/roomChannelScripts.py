import cv2
import numpy as np

#closet channel
def rchannel1(img):
  h_im, w_im = img.shape[0],img.shape[1]
  canvas = np.full((h_im,w_im),0)
  for i in range(h_im):
    for j in range(w_im):
      if  tuple(img[i][j]) == (192,192,224):
        canvas[i][j] = 1
  return canvas

#bathroom/washroom channel
def rchannel2(img):
  h_im, w_im = img.shape[0],img.shape[1]
  canvas = np.full((h_im,w_im),0)
  for i in range(h_im):
    for j in range(w_im):
      if  tuple(img[i][j]) == (192,255,255):
        canvas[i][j] = 1
  return canvas

#living/dining/kitchen channel
def rchannel3(img):
  h_im, w_im = img.shape[0],img.shape[1]
  canvas = np.full((h_im,w_im),0)
  for i in range(h_im):
    for j in range(w_im):
      if  tuple(img[i][j]) == (224,255,192):
        canvas[i][j] = 1
  return canvas

#bedroom channel
def rchannel4(img):
  h_im, w_im = img.shape[0],img.shape[1]
  canvas = np.full((h_im,w_im),0)
  for i in range(h_im):
    for j in range(w_im):
      if  tuple(img[i][j]) == (255,224,128):
        canvas[i][j] = 1
  return canvas

#hall channel
def rchannel5(img):
  h_im, w_im = img.shape[0],img.shape[1]
  canvas = np.full((h_im,w_im),0)
  for i in range(h_im):
    for j in range(w_im):
      if  tuple(img[i][j]) == (255,160, 96):
        canvas[i][j] = 1
  return canvas

#balcony channel
def rchannel6(img):
  h_im, w_im = img.shape[0],img.shape[1]
  canvas = np.full((h_im,w_im),0)
  for i in range(h_im):
    for j in range(w_im):
      if  tuple(img[i][j]) == (255,224,224):
        canvas[i][j] = 1
  return canvas

#balcony channel
def rchannel7(img):
  h_im, w_im = img.shape[0],img.shape[1]
  canvas = np.full((h_im,w_im),0)
  for i in range(h_im):
    for j in range(w_im):
      if  tuple(img[i][j]) == (224,224,128):
        canvas[i][j] = 1
  return canvas

#background channel
def rchannel8(img):
  h_im, w_im = img.shape[0],img.shape[1]
  canvas = np.full((h_im,w_im),1)
  col_list = [(192,192,224),(192,255,255),(224,255,192),(255,224,128),(255,160, 96),(255,224,224),(224,224,128)]
  for i in range(h_im):
    for j in range(w_im):
      if  tuple(img[i][j]) in col_list:
        canvas[i][j] = 0
  return canvas

#resize individual channel
#special case when background channel
def preprocess_room_mask_channel(img, ischannel7=False):
  img = img.astype('float32')
  #create empty canvas
  h_out = w_out = 512
  canvas = np.full((h_out,w_out),0)
  if ischannel7 == True:
    canvas = np.full((h_out,w_out),1)
  h_im, w_im = img.shape[0],img.shape[1]
  #find resized image height, image width
  newh = int(h_im * min(h_out/h_im, w_out/w_im))
  neww = int(w_im * min(h_out/h_im, w_out/w_im))
  #layout on canvas
  img_rs = cv2.resize(img,(neww,newh),interpolation=cv2.INTER_LINEAR)
  canvas[(h_out-newh)//2:(h_out-newh)//2 + newh, (w_out-neww)//2:(w_out-neww)//2 + neww] = img_rs
  return canvas