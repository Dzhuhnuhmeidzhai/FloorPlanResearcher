import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Conv2DTranspose

def encoding_type1(inp, filters):
    x = Conv2D(filters=filters, kernel_size=(3,3), activation='relu', padding='same')(inp)
    x = Conv2D(filters=filters, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    return x

def encoding_type2(inp, filters):
    x = Conv2D(filters=filters, kernel_size=(3,3), activation='relu', padding='same')(inp)
    x = Conv2D(filters=filters, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = Conv2D(filters=filters, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    return x

def decoding(inp1,inp2,filters):
    x = Conv2DTranspose(filters=filters, kernel_size=(4,4), activation='linear', padding='same', strides=2)(inp1)
    y = Conv2D(filters=filters, kernel_size=(1,1), activation='linear', padding='same')(inp2)
    z = x+y
    z = Conv2D(filters=filters, kernel_size=(3,3), activation='linear', padding='same')(z)
    return z

def final_decoding(inp, filters):
    x = Conv2D(filters=filters, kernel_size=(1,1), activation='relu', padding='same')(inp)
    x = tensorflow.image.resize(x, (512,512))
    return x

def complex_decode(inp1,inp2,filters):
    stride = 4
    #upper layer connection
    x = Conv2D(filters=filters, kernel_size=(3,3), activation='relu', padding='same')(inp1)
    x = Conv2D(filters=filters, kernel_size=(3,3), activation='relu', padding='same')(x)
    a = Conv2D(filters=1, kernel_size=(1,1), activation='relu', padding='same')(x)
    a = tensorflow.keras.activations.sigmoid(a)
    #lower layer
    y = Conv2D(filters=filters, kernel_size=(3,3), activation='relu', padding='same')(inp2)
    b = Conv2D(filters=1, kernel_size=(1,1), activation='relu', padding='same')(y)
    #first attention
    b = tensorflow.math.multiply(a, b)
    #height, width calculation for direction aware kernels
    height = b.shape[1]
    width = b.shape[2]
    hs = height//stride
    ws = width//stride
    #horizontal direction-aware kernel
    kernel_h = tensorflow.ones((hs,1,1,1))
    b_h = tensorflow.keras.backend.conv2d(b,kernel_h,strides=(1,1,1,1),padding='same')
    #vertical direction-aware kernel
    kernel_v = tensorflow.ones((1,ws,1,1))
    b_v = tensorflow.keras.backend.conv2d(b,kernel_v,strides=(1,1,1,1),padding='same')
    #diagonal1 direction-aware kernel
    kernel_d1 = tensorflow.eye(hs,ws)
    kernel_d1 = tensorflow.reshape(kernel_d1,(hs,ws,1,1))
    b_d1 = tensorflow.keras.backend.conv2d(b,kernel_d1,strides=(1,1,1,1),padding='same')
    #diagonal1 direction-aware kernel
    kernel_d1 = tensorflow.eye(hs,ws)
    kernel_d1 = tensorflow.reshape(kernel_d1,(hs,ws,1,1))
    b_d1 = tensorflow.keras.backend.conv2d(b,kernel_d1,strides=(1,1,1,1),padding='same')
    #diagonal2 direction-aware kernel
    kernel_d2 = tensorflow.eye(hs,ws)
    kernel_d2 = tensorflow.reshape(kernel_d2,(hs,ws,1))
    kernel_d2 = tensorflow.image.flip_left_right(kernel_d2)
    kernel_d2 = tensorflow.reshape(kernel_d2,(hs,ws,1,1))
    b_d2 = tensorflow.keras.backend.conv2d(b,kernel_d2,strides=(1,1,1,1),padding='same')
    #second attention
    dir_aware_kernel = b_h + b_v + b_d1 + b_d2
    c = tensorflow.math.multiply(a, dir_aware_kernel)
    #expand the second attention results
    c = Conv2D(filters=filters, kernel_size=(1,1), activation='relu', padding='same')(c)
    complex_out = tensorflow.keras.backend.concatenate((inp2,c), axis=-1)
    complex_out = Conv2D(filters=filters, kernel_size=(3,3), activation='relu', padding='same')(complex_out)
    return complex_out