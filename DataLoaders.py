import numpy as np
import tensorflow as tf
import imgaug.augmenters as iaa
import math

'''
Work in Progress: Not Finished

https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

Useful for storing datasets on the disk!

'''
class standard_aug_loader(tf.keras.utils.Sequence):
  def __init__(self, x, y, aug, batch_size):
    self.x, self.y = x, y
    self.aug = aug
    self.batch_size = batch_size
    
  def __getitem__(self, idx):
    batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
    batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]
    batch_x = aug(images=batch_x)
    return batch_x, batch_y

class org_aug_pair_loader(tf.keras.utils.Sequence):
  def __init__(self, x, y, aug, batch_size):
    self.x, self.y = x, y
    self.aug = aug
    self.batch_size = batch_size
    
  def __getitem__(self, idx):
    batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
    batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]
    aug_batch = aug(images=batch_x)
    return batch_x, aug_batch, batch_y

class org_aug_randaug_triple_loader(tf.keras.utils.Sequence):
  def __init__(self, x, y, randaug, aug, dim=(32,32), n_channels=3, batch_size):
    self.x, self.y = x, y
    self.aug = aug
    self.batch_size = batch_size
    self.shuffle = True
    self.on_epoch_end()
  
  def __getitem__(self, idx):
    batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
    batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]
    randaug_batch = randaug(images=batch_x)
    paired_aug_batch = aug(images=batch_x)
    return batch_x, paired_aug_batch, randaug_batch, batch_y
 
  def on_epoch_end(self):
    # Updates indexes after each epoch
    self.indexes = np.arange(len(self.x))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)
      
  def __data_generation(self, list_IDs_temp):
    # Generates data containing batch_size samples
    # X : (n_samples, *dim, n_channels)
    X = np.empty((
