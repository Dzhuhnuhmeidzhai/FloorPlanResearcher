import numpy as np
import tensorflow.keras

class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, train_input, train_target, batch_size=1, dim=(512,512), n_channels=11):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.train_target = train_target
        self.train_input = train_input
        self.n_channels = n_channels
        self.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.train_input) / self.batch_size))
    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X = self.train_input[indexes[0]]
        y = self.train_target[indexes[0]]
        return X, y
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.train_input))