import getImagePaths
from getImagePaths import *
import loadingImages
from loadingImages import *


#get training images
def getTrainImages(input_dir = "Train/Train/Raw/"
    ,target_dir_w = "Train/Train/Walls/"
    ,target_dir_dw = "Train/Train/DoorsAndWindows/"
    ,target_dir_r = "Train/Train/Rooms/"):
    #input_dir: directory where raw train images are stored
    #target_dir_w: directory where train walls-annotated images are stored
    #target_dir_dw: directory where train doors/windows-annotated images are stored
    #target_dir_r: directory where train rooms-annotated images are stored
    #----ALL OF THE ABOVE ARE SAME SIZE/DIMENSIONS AS THE RAW IMAGE----#
    #----ABOVE IMAGES WILL BE USED TO CREATE TRAIN INPUTS AND TARGET MASKS----#
    ##############################
    #get training images' paths
    i_p, iw_p, idw_p, ir_p = getImagePaths(input_dir,target_dir_w, target_dir_dw, target_dir_r)
    #list of training images
    input_images = []
    walls_images = []
    doorswindows_images = []
    rooms_images = []
    #get training images
    for i in range(len(i_p)):
      im, w_im, dw_im, r_im = loadingImages(i_p[i],iw_p[i],idw_p[i],ir_p[i])
      input_images.append(im)
      walls_images.append(w_im)
      doorswindows_images.append(dw_im)
      rooms_images.append(r_im)
    return input_images, walls_images, doorswindows_images, rooms_images