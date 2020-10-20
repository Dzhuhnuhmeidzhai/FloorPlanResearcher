import resizeInput
from resizeInput import *

def getInputMasks(input_images):
	#placeholder for resized input images
	resized_imasks = []
	print("Serial no. of input image being processed: ")
	#resize input images
	for i in range(len(input_images)):
		print(i)
		resized_i = resizeInput(input_images[i])
		resized_imasks.append(resized_i)
	return resized_imasks


	
