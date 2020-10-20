import tensorflow


def customMetric(y_true, y_pred):
  #get ground truth channels
  nbd_true = y_true[:,:,:,0]
  ndw_true = y_true[:,:,:,1]
  nw_true = y_true[:,:,:,2]
  #get predicted channels
  nbd_pred = y_pred[:,:,:,0]
  ndw_pred = y_pred[:,:,:,1]
  nw_pred = y_pred[:,:,:,2]
  #compare ground-truth and predicted channels
  nbd_cmp = tensorflow.equal(nbd_true,nbd_pred)
  ndw_cmp = tensorflow.equal(ndw_true,ndw_pred)
  nw_cmp = tensorflow.equal(nw_true,nw_pred)
  #count number of correct pixel-classifications
  nbd_cor = tensorflow.math.count_nonzero(nbd_cmp)
  ndw_cor = tensorflow.math.count_nonzero(ndw_cmp)
  nw_cor = tensorflow.math.count_nonzero(nw_cmp)
  #define accuracy
  acc = (nbd_cor + ndw_cor + nw_cor)/(3*512*512)
  return acc