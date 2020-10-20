import tensorflow 
import tensorflow.keras
import customLoss1
from customLoss1 import *
import customLoss2
from customLoss2 import *

#common customLoss function
def customLoss(y_true,y_pred):
	#extract boundaries part
	bd_part_true = y_true[:,:,:,0:3]
	bd_part_pred = y_pred[:,:,:,0:3]
	#extract rooms part
	rm_part_true = y_true[:,:,:,3:]
	rm_part_pred = y_pred[:,:,:,3:]
	#get boundary loss
	l_bt, n_bt = customLoss1(bd_part_true,bd_part_pred)
	#get room loss 
	l_rt, n_rt = customLoss2(rm_part_true,rm_part_pred)
	#get loss weights
	w_bt = n_rt/(n_bt + n_rt)
	w_rt = n_bt/(n_bt + n_rt)
	#compute total loss
	total_loss = (w_bt * l_bt) + (w_rt * l_rt)
	return total_loss