import os
import getContours
from getContours import *

def saveTrainContours(base_path = r'drive/My Drive/CCTech/Model/BoundariesBranch/Predictions/Train/Predicted/'
                      ,train_cont_dir = r'drive/My Drive/CCTech/Contours/Train/ContourImages/'
                      ,train_info_dir = r'drive/My Drive/CCTech/Contours/Train/ContourInfo/'):
  #base path: directory where predicted boundaries of training data are stored
  #train_cont_dir: directory where you want to store contours' images of predicted boundaries of train data
  #train_info_dir: directory where you want to store info of contours of predicted boundaries of train data
  ##############################
  #get images' names
  l_ims = sorted(os.listdir(base_path))
  #get contours
  print("Serial number of training data boundary prediction whose contours are being computed:")
  for i in l_ims:
    print(i)
    path = base_path + i
    getContours(path, train_cont_dir, train_info_dir)