import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import paperLayers
from paperLayers import *
import customLoss
from customLoss import *
import customMetric
from customMetric import *

def paperLayersModel():
	#encoding blocks
	input_layer = Input(shape=(512, 512, 3))
	c1 = encoding_type1(input_layer, filters=64)
	c2 = encoding_type1(c1, filters=128)
	c3 = encoding_type2(c2, filters=256)
	c4 = encoding_type2(c3, filters=512)
	c5 = encoding_type2(c4, filters=512)
	#decoding blocks
	u1 = decoding(c5,c4,256)
	u2 = decoding(u1,c3,128)
	u3 = decoding(u2,c2,64)
	u4 = decoding(u3,c1,32)
	u_final = final_decoding(u4,3)

	#softmax
	u_out = tensorflow.keras.activations.softmax(u_final)
	#model details
	model = Model(inputs=input_layer, outputs=u_out)
	model.compile(optimizer='adam',loss=customLoss, metrics=[customMetric])
	model.summary()

	return model