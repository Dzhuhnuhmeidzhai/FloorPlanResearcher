import tensorflow as tf
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
    x = Conv2D(filters=filters, kernel_size=(3,3), activation='relu', padding='same')(inp)
    x = tf.image.resize(x, (512,512))
    return x