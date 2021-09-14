import keras
import numpy as np
from tensorflow.keras import layers
input_shape = (32,32,3)
from tensorflow.keras.applications import ResNet152V2
resnet = ResNet152V2(weights=None, input_shape=input_shape)

def standard_model(x_train):
  normalization_layer = keras.Sequential(
    [
      layers.experimental.preprocessing.Normalization(),
    ],
    name="normalization",
  )
  normalization_layer.layers[0].adapt(x_train)
  
  inputs = layers.Input(shape=input_shape)
  normalized = normalization_layer(inputs)
  resnet_outputs = ResNet152V2(include_top=False, weights=None)(normalized)
  flattened = layers.Flatten()(resnet_outputs)
  dense_1 = layers.Dense(512, activation="relu")(flattened)
  dense_2 = layers.Dense(512, activation="relu")(dense_1)
  outputs = layers.Dense(10, activation="softmax")(dense_2)
  model = keras.Model(inputs=inputs, outputs=outputs)
  return model

class Aug_Multiplicity_Model(keras.Model):
  def __init__(self, model):
    super(Aug_Multiplicity_Model, self).__init__()
    self.model = model

  def train_step(self, data):
    [org_x, aug_1, aug_2, aug_3, aug_4], y = data
    '''
    # re-arrange this so you zip the data and parse it here
    # e.g. org_x, y = org
    # e.g. aug_1, aug_2, aug_3, aug_4 = augs
    '''
    with tf.GradientTape() as tape:
        org_pred = self(org_x, training=True)
        aug_pred_1 = self(aug_1, training=True)
        aug_pred_2 = self(aug_2, training=True)
        aug_pred_3 = self(aug_3, training=True)
        aug_pred_4 = self(aug_4, training=True)

        loss = self.compiled_loss(y, org_pred,
                                   regularization_losses=self.losses)
        loss += self.compiled_loss(y, aug_pred_1,
                                   regularization_losses=self.losses)
        loss += self.compiled_loss(y, aug_pred_2,
                                   regularization_losses=self.losses)
        loss += self.compiled_loss(y, aug_pred_3,
                                   regularization_losses=self.losses)
        loss += self.compiled_loss(y, aug_pred_4,
                                   regularization_losses=self.losses)

    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    self.compiled_metrics.update_state(y, org_pred)
    return {m.name: m.result() for m in self.metrics}
  
  def call(self, data):
    return self.model(data)
  
def aug_multiplicity_model(x_train):
  model = standard_model(x_train)
  return Aug_Multiplicity_Model(model)

class Consistency_Model(keras.Model):
  def __init__(self, model):
    super(Consistency_Model, self).__init__()
    self.model = model

  def train_step(self, data):
    [org_data, aug_pair], y = data

    with tf.GradientTape() as tape:
      y_pred = self(org_data, training=True)

      aug_pred = self(aug_pair, training=True)

      loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
      # maybe want to re-weight these
      loss += self.compiled_loss(y_pred, aug_pred, regularization_losses=self.losses)

    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    self.compiled_metrics.update_state(y, y_pred)
    return {m.name: m.result() for m in self.metrics}
  
  def call(self, data):
    return self.model(data)
  
def consistency_loss_model(x_train):
  model = standard_model(x_train)
  return Consistency_Model(model)

