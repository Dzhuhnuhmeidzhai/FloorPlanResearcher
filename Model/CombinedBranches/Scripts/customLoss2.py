import tensorflow 
import tensorflow.keras
bce = tensorflow.keras.losses.BinaryCrossentropy()


def customLoss2(y_true, y_pred):
  #count ground-truth pixels of each class
  n_closet = 0
  n_wash = 0
  n_liv = 0
  n_bed = 0
  n_hall = 0
  n_bal = 0
  n_ot = 0
  n_bg = 0
  n_closet = tensorflow.math.count_nonzero(y_true[:,:,:,0])
  n_wash = tensorflow.math.count_nonzero(y_true[:,:,:,1])
  n_liv = tensorflow.math.count_nonzero(y_true[:,:,:,2])
  n_bed = tensorflow.math.count_nonzero(y_true[:,:,:,3])
  n_hall = tensorflow.math.count_nonzero(y_true[:,:,:,4])
  n_bal = tensorflow.math.count_nonzero(y_true[:,:,:,5])
  n_ot = tensorflow.math.count_nonzero(y_true[:,:,:,6])
  n_bg = tensorflow.math.count_nonzero(y_true[:,:,:,7])
  #total pixels
  n = n_closet + n_wash + n_liv + n_bed + n_hall + n_bal + n_ot + n_bg
  #total room type pixels
  n_rt_px = n_closet + n_wash + n_liv + n_bed + n_hall + n_bal + n_ot
  #class weights
  w_closet = (n - n_closet)/((n-n_closet) + (n-n_wash) + (n-n_liv) + (n-n_bed) + (n-n_hall) + (n-n_bal) + (n-n_ot) + (n-n_bg))
  w_wash = (n - n_wash)/((n-n_closet) + (n-n_wash) + (n-n_liv) + (n-n_bed) + (n-n_hall) + (n-n_bal) + (n-n_ot) + (n-n_bg))
  w_liv = (n - n_liv)/((n-n_closet) + (n-n_wash) + (n-n_liv) + (n-n_bed) + (n-n_hall) + (n-n_bal) + (n-n_ot) + (n-n_bg))
  w_bed = (n - n_bed)/((n-n_closet) + (n-n_wash) + (n-n_liv) + (n-n_bed) + (n-n_hall) + (n-n_bal) + (n-n_ot) + (n-n_bg))
  w_hall = (n - n_hall)/((n-n_closet) + (n-n_wash) + (n-n_liv) + (n-n_bed) + (n-n_hall) + (n-n_bal) + (n-n_ot) + (n-n_bg))
  w_bal = (n - n_bal)/((n-n_closet) + (n-n_wash) + (n-n_liv) + (n-n_bed) + (n-n_hall) + (n-n_bal) + (n-n_ot) + (n-n_bg))
  w_ot = (n - n_ot)/((n-n_closet) + (n-n_wash) + (n-n_liv) + (n-n_bed) + (n-n_hall) + (n-n_bal) + (n-n_ot) + (n-n_bg))
  w_bg = (n - n_bg)/((n-n_closet) + (n-n_wash) + (n-n_liv) + (n-n_bed) + (n-n_hall) + (n-n_bal) + (n-n_ot) + (n-n_bg))
  weights = [w_closet, w_wash, w_liv, w_bed, w_hall, w_bal, w_ot, w_bg]
  y1 = tensorflow.reshape(y_true,(512,512,8))
  y2 = tensorflow.reshape(y_pred,(512,512,8))
  y3 = tensorflow.cast(y2, dtype = tensorflow.float64)
  #calculate loss
  loss = 0
  for cls in range(0,7):
        loss= loss + ((weights[cls])*(bce(y1[:,:,cls],y3[:,:,cls])))
  return loss, n_rt_px