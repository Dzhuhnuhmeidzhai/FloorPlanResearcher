import tensorflow 
import tensorflow.keras
bce = tensorflow.keras.losses.BinaryCrossentropy()


def customLoss1(y_true, y_pred):
  #count ground-truth pixels of each class
  nbd = 0
  ndw = 0
  nw = 0
  nbd = tensorflow.math.count_nonzero(y_true[:,:,:,0])
  ndw = tensorflow.math.count_nonzero(y_true[:,:,:,1])
  nw = tensorflow.math.count_nonzero(y_true[:,:,:,2])
  #total pixels
  n = nbd + ndw + nw
  #total boundary pixels
  n_bd_px = ndw + nw
  #class weights
  w_bd = (n - nbd)/((n-nbd) + (n-ndw) + (n-nw))
  w_dw = (n - ndw)/((n-nbd) + (n-ndw) + (n-nw))
  w_w = (n - nw)/((n-nbd) + (n-ndw) + (n-nw))
  weights = [w_bd, w_dw, w_w]
  y1 = tensorflow.reshape(y_true,(512,512,3))
  y2 = tensorflow.reshape(y_pred,(512,512,3))
  y3 = tensorflow.cast(y2, dtype = tensorflow.float64)
  #calculate loss
  loss = 0
  for cls in range(0,3):
        loss= loss + ((weights[cls])*(bce(y1[:,:,cls],y3[:,:,cls])))
  return loss, n_bd_px