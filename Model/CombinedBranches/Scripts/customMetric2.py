import tensorflow


def customMetric2(y_true, y_pred):
  y_true_rm = y_true[:,:,:,3:]
  y_pred_rm = y_pred[:,:,:,3:]
  #get ground truth channels
  ncloset_true = y_true_rm[:,:,:,0]
  nwash_true = y_true_rm[:,:,:,1]
  nliv_true = y_true_rm[:,:,:,2]
  nbed_true = y_true_rm[:,:,:,3]
  nhall_true = y_true_rm[:,:,:,4]
  nbal_true = y_true_rm[:,:,:,5]
  not_true = y_true_rm[:,:,:,6]
  nbg_true = y_true_rm[:,:,:,7]
  #get predicted channels
  ncloset_pred = y_pred_rm[:,:,:,0]
  nwash_pred = y_pred_rm[:,:,:,1]
  nliv_pred = y_pred_rm[:,:,:,2]
  nbed_pred = y_pred_rm[:,:,:,3]
  nhall_pred = y_pred_rm[:,:,:,4]
  nbal_pred = y_pred_rm[:,:,:,5]
  not_pred = y_pred_rm[:,:,:,6]
  nbg_pred = y_pred_rm[:,:,:,7]
  #compare ground-truth and predicted channels
  ncloset_cmp = tensorflow.equal(ncloset_true,ncloset_pred)
  nwash_cmp = tensorflow.equal(nwash_true,nwash_pred)
  nliv_cmp = tensorflow.equal(nliv_true,nliv_pred)
  nbed_cmp = tensorflow.equal(nbed_true,nbed_pred)
  nhall_cmp = tensorflow.equal(nhall_true,nhall_pred)
  nbal_cmp = tensorflow.equal(nbal_true,nbal_pred)
  not_cmp = tensorflow.equal(not_true,not_pred)
  nbg_cmp = tensorflow.equal(nbg_true,nbg_pred)
  #count number of correct pixel-classifications
  ncloset_cor = tensorflow.math.count_nonzero(ncloset_cmp)
  nwash_cor = tensorflow.math.count_nonzero(nwash_cmp)
  nliv_cor = tensorflow.math.count_nonzero(nliv_cmp)
  nbed_cor = tensorflow.math.count_nonzero(nbed_cmp)
  nhall_cor = tensorflow.math.count_nonzero(nhall_cmp)
  nbal_cor = tensorflow.math.count_nonzero(nbal_cmp)
  not_cor = tensorflow.math.count_nonzero(not_cmp)
  nbg_cor = tensorflow.math.count_nonzero(nbg_cmp)
  #define accuracy
  acc = (ncloset_cor + nwash_cor + nliv_cor + nbed_cor + nhall_cor + nbal_cor + not_cor + nbg_cor)/(8*512*512)
  return acc