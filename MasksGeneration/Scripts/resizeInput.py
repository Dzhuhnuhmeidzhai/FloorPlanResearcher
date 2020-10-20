import numpy as np
import cv2


#resizing of raw input image
def resizeInput(img):
  #create empty canvas
  h_out = w_out = 512
  canvas = np.full((h_out,w_out,3),255)
  h_im, w_im = img.shape[0],img.shape[1]
  #find resized image height, image width
  newh = int(h_im * min(h_out/h_im, w_out/w_im))
  neww = int(w_im * min(h_out/h_im, w_out/w_im))
  #layout on canvas
  img_rs = cv2.resize(img,(neww,newh),interpolation=cv2.INTER_LINEAR)
  canvas[(h_out-newh)//2:(h_out-newh)//2 + newh, (w_out-neww)//2:(w_out-neww)//2 + neww, :] = img_rs
  return canvas 