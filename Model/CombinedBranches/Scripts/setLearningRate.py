import tensorflow as tf
import tensorflow.keras

#define LR based on epoch condition
def scheduler(epoch):
	if epoch < 10:
		return 0.0001
	elif epoch < 100:
		return 0.00001
	else:
		return 0.000001

#set LR
def setLR():
	callback = tensorflow.keras.callbacks.LearningRateScheduler(scheduler)
	return callback
