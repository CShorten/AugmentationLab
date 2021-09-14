import numpy as np
import tensorflow as tf
import imgaug.augmenters as iaa
import math

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
