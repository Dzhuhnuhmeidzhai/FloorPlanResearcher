import getImagePaths
from getImagePaths import *
import loadingImages
from loadingImages import *


#get testing images
def getTestImages(input_dir = "Test/Test/Raw/"
	,target_dir_w = "Test/Test/Walls/"
	,target_dir_dw = "Test/Test/DoorsAndWindows/"
	,target_dir_r = "Test/Test/Rooms/"):
    #input_dir: directory where raw test images are stored
    #target_dir_w: directory where test walls-annotated images are stored
    #target_dir_dw: directory where test doors/windows-annotated images are stored
    #target_dir_r: directory where test rooms-annotated images are stored
    #----ALL OF THE ABOVE ARE SAME SIZE/DIMENSIONS AS THE RAW IMAGE----#
    #----ABOVE IMAGES WILL BE USED TO CREATE TEST INPUTS AND TARGET MASKS----#
    ##############################
    #get testing images' paths
    i_p, iw_p, idw_p, ir_p = getImagePaths(input_dir,target_dir_w, target_dir_dw, target_dir_r)
    #list of testing images
    input_images = []
    walls_images = []
    doorswindows_images = []
    rooms_images = []
    #get testing images
    for i in range(len(i_p)):
      im, w_im, dw_im, r_im = loadingImages(i_p[i],iw_p[i],idw_p[i],ir_p[i])
      input_images.append(im)
      walls_images.append(w_im)
      doorswindows_images.append(dw_im)
      rooms_images.append(r_im)
    return input_images, walls_images, doorswindows_images, rooms_images