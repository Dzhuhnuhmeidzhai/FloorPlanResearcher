import cv2
import tensorflow
import tensorflow.keras
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology, img_as_bool, segmentation, io
from scipy import ndimage as ndi
import os

def getContours(imname, cont_dir, info_dir):
  #get image
  im = cv2.imread(imname)
  im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  #get walls, doorsWindows combined channel
  c1c2im = im_rgb[:,:,1] | im_rgb[:,:,2]
  #thin combined image
  c1c2_thin = cv2.ximgproc.thinning(c1c2im)
  #fill gaps using distance transform
  out = ndi.distance_transform_edt(~c1c2_thin)
  out = out < 0.06 * out.max()
  out = morphology.skeletonize(out)
  #get combined, thinned, filled image
  x = np.zeros((512,512))
  for i in range(512):
    for j in range(512):
      if out[i][j]==True:
        x[i][j] = 255
      elif out[i][j]==False:
        x[i][j] = 0
  #convert to uint8 for openCV
  x8 = np.asarray(x, dtype=np.uint8)
  #find contours
  (contours,_) = cv2.findContours(x8,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  #manage contours
  len_contours = len(contours)
  final_contours = []
  for i in range(len_contours):
    #clean useless contours
    if len(contours[i]) <= 2:
      continue
    else:
      final_contours.append(contours[i])
      #contour on clean slate
      temp_im = np.zeros((512,512))
      ith_contour = cv2.drawContours(temp_im.copy(), contours, i, (255,255,255), 3)
      #save ith contour image
      final_path = cont_dir + imname.split('/')[-1].split('.')[0] + '_' + str(i) + '.jpg'
      wr1 = cv2.imwrite(final_path, ith_contour)
  #save all contours combined
  all_contours = cv2.drawContours(temp_im.copy(), final_contours, -1, (255,255,255), 3)
  all_path = cont_dir + imname.split('/')[-1].split('.')[0] + '_all.jpg'
  wr2 = cv2.imwrite(all_path, all_contours)
  #save contours information
  coords_file = info_dir + imname.split('/')[-1].split('.')[0] + '_contours.txt'
  area_file = info_dir + imname.split('/')[-1].split('.')[0] + '_area.txt'
  with open(coords_file, 'w') as f:
    for item in final_contours:
        f.write("%s\n" % item)
  with open(area_file, 'w') as f:
    for item in final_contours:
        area = cv2.contourArea(item)
        f.write("%s\n" % area)