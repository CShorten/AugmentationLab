import keras
import tensorflow as tf

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
  
  
'''
A more general class design is to just have the two augs as arguments
i.e. randaug, rotate or randaug, randaug ... rotate, rotate ... crop, rotate ...
'''
class Consistency_Model_with_RandAug(keras.Model):
  def __init__(self, model, consistency_weight, org_matching, aug_grads):
    super(Consistency_Model_with_RandAug, self).__init__()
    self.model = model
    self.consistency_weight = consistency_weight
    self.org_matching = org_matching
    self.aug_grads = aug_grads

  def train_step(self, data):
    [randaug_x, org_x, [aug_xs]], y = data # change this so you can pass in a variable number of augmented xs

    with tf.GradientTape() as tape:
      # Cross Entropy loss between RandAug Prediction and Ground Truth Y Label
      randaug_pred = self(randaug_x, training=True)
      loss = self.compiled_loss(y, randaug_pred, regularization_losses=self.losses)
      
      # Consistency loss
      if org_matching==True:
        matching_pred = self(org_x, training=True)
      else:
        matching_pred = randaug_pred
      
      for aug_x in aug_xs:
        aug_pred = self(aug_x, training=self.aug_grads)
        loss += self.consistency_weight * self.compiled_loss(org_y_pred, aug_pred, regularization_losses=self.losses) # todo add fine-grained loss weightings

    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    self.compiled_metrics.update_state(y, randaug_pred)
    return {m.name: m.result() for m in self.metrics}
  
  def call(self, data):
    return self.model(data)
  
# Not sure how useful this wrapper is...
def consistency_loss_model(x_train):
  model = standard_model(x_train)
  return Consistency_Model(model)

# Second class so you can pass in the intermediate models
# Need to add the RandAugment, y into this
class Deep_Consistency_Model(keras.Model):
  def __init__(self, model, intermediate_layer_models):
    super(Deep_Consistency_Model, self).__init__()
    self.model = model
    self.intermediate_layer_models = intermediate_layer_models
    self.num_rep_layers = len(self.intermediate_layer_models)

  def train_step(self, data):
    [org_data, aug_pair], y = data

    with tf.GradientTape() as tape:
      y_pred = self(org_data, training=True)

      aug_pred = self(aug_pair, training=True)

      loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
      
      # Consistency Losses
      loss += self.compiled_loss(y_pred, aug_pred, regularization_losses=self.losses)

      for i in range(self.num_rep_layers):
        org_med = self.intermediate_layer_models[i](org_data, training=True)
        aug_med = self.intermediate_layer_models[i](aug_pair, training=True)
        loss += self.compiled_loss(org_med, aug_med, regularization_losses=self.losses)
        
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    self.compiled_metrics.update_state(y, y_pred)
    return {m.name: m.result() for m in self.metrics}
  
  def call(self, data):
    return self.model(data)
