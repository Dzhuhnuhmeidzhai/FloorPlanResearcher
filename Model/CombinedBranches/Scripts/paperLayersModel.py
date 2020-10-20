import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
import paperLayers
from paperLayers import *
import customLoss
from customLoss import *
import customMetric1
from customMetric1 import *
import customMetric2
from customMetric2 import *

def paperLayersModel():
	#encoding blocks
	input_layer = Input(shape=(512, 512, 3))
	c1 = encoding_type1(input_layer, filters=64)
	c2 = encoding_type1(c1, filters=128)
	c3 = encoding_type2(c2, filters=256)
	c4 = encoding_type2(c3, filters=512)
	c5 = encoding_type2(c4, filters=512)
	#upper decoding blocks
	u1 = decoding(c5,c4,256)
	u2 = decoding(u1,c3,128)
	u3 = decoding(u2,c2,64)
	u4 = decoding(u3,c1,32)
	u_final = final_decoding(u4,3)
  #lower decoding blocks
	l1 = decoding(c5,c4,256)
	l1_c = complex_decode(u1,l1,256)
	l2 = decoding(l1_c,c3,128)
	l2_c = complex_decode(u2,l2,128)
	l3 = decoding(l2_c,c2,64)
	l3_c = complex_decode(u3,l3,64)
	l4 = decoding(l3_c,c1,32)
	l4_c = complex_decode(u4,l4,32)
	l_final = final_decoding(l4_c,8)
	#softmax
	u_out = tensorflow.keras.activations.softmax(u_final)
	l_out = tensorflow.keras.activations.softmax(l_final)
	out = tensorflow.concat([u_out,l_out],axis=-1)
	out_cust = Lambda(lambda x:x, name = "out")(out)
	#model details
	model = Model(inputs=input_layer, outputs=out_cust)
	model.compile(optimizer='adam',loss=customLoss, metrics={"out":[customMetric1,customMetric2]})
	model.summary()

	return model