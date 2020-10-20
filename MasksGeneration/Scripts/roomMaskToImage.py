import numpy as np

#room image from room mask
def mask2visible(mask):
  visible = np.zeros((512,512,3))
  #get channels
  c1 = mask[:,:,0]
  c2 = mask[:,:,1]
  c3 = mask[:,:,2]
  c4 = mask[:,:,3]
  c5 = mask[:,:,4]
  c6 = mask[:,:,5]
  c7 = mask[:,:,6]
  #color pixels accordingly
  for i in range(512):
    for j in range(512):
      if c1[i][j] == 1:
        visible[i][j] = [192,192,224]
      if c2[i][j] == 1:
        visible[i][j] = [192,255,255]
      if c3[i][j] == 1:
        visible[i][j] = [224,255,192]
      if c4[i][j] == 1:
        visible[i][j] = [255,224,128]
      if c5[i][j] == 1:
        visible[i][j] = [255,160, 96]
      if c6[i][j] == 1:
        visible[i][j] = [255,224,224]
      if c7[i][j] == 1:
        visible[i][j] = [224,224,128]
  return visible
