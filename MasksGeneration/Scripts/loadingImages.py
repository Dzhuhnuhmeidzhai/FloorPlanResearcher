import tensorflow 
import tensorflow.keras
import tensorflow.keras.preprocessing


#load images from image paths
def loadingImages(inp_path, wmask_path, dwmask_path, rmask_path):
    #load image from path
    input_image = tensorflow.keras.preprocessing.image.load_img(inp_path)
    w_image = tensorflow.keras.preprocessing.image.load_img(wmask_path)
    dw_image = tensorflow.keras.preprocessing.image.load_img(dwmask_path)
    r_image = tensorflow.keras.preprocessing.image.load_img(rmask_path)
    #convert images to tensors
    input_image = tensorflow.keras.preprocessing.image.img_to_array(input_image)
    w_image = tensorflow.keras.preprocessing.image.img_to_array(w_image)
    dw_image = tensorflow.keras.preprocessing.image.img_to_array(dw_image)
    r_image = tensorflow.keras.preprocessing.image.img_to_array(r_image)
    return input_image, w_image, dw_image, r_image