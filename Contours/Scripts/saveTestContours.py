import os
import getContours 
from getContours import *

def saveTestContours(base_path = r'drive/My Drive/CCTech/Model/BoundariesBranch/Predictions/Test/Predicted/'
                     ,test_cont_dir = r'drive/My Drive/CCTech/Contours/Test/ContourImages/'
                     ,test_info_dir = r'drive/My Drive/CCTech/Contours/Test/ContourInfo/'):
  #base path: directory where predicted boundaries of testing data are stored
  #test_cont_dir: directory where you want to store contours' images of predicted boundaries of test data
  #test_info_dir: directory where you want to store info of contours of predicted boundaries of test data
  ##############################
  #get images' names
  l_ims = sorted(os.listdir(base_path))
  #get contours
  print("Serial number of test data boundary prediction whose contours are being computed:")
  for i in l_ims:
    print(i)
    path = base_path + i
    getContours(path, test_cont_dir, test_info_dir)