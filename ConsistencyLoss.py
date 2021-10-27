import keras
import tensorflow as tf

class Consistency_Model(keras.Model):
  def __init__(self, model, consistency_weight, org_matching, aug_grads,
              intermediate_layer_matching=False, intermediate_layer_model=None):
    super(Consistency_Model, self).__init__()
    self.model = model
    self.consistency_weight = consistency_weight
    self.org_matching = org_matching
    self.aug_grads = aug_grads
    self.intermediate_layer_matching = intermediate_layer_matching
    self.intermediate_layer_model = intermediate_layer_model

  def train_step(self, data):
    # figure out how to pass in a list of aug_xs
    [randaug_x, org_x, aug_x], y = data # change this so you can pass in a variable number of augmented xs

    with tf.GradientTape() as tape:
      # Cross Entropy loss between RandAug Prediction and Ground Truth Y Label
      randaug_pred = self(randaug_x, training=True)
      loss = self.compiled_loss(y, randaug_pred, regularization_losses=self.losses)
      
      # Consistency loss
      if (self.intermediate_layer_matching == True): # Vector Representation Consistency
        if (self.org_matching == True):
          matching_data = org_x
        else:
          matching_data = randaug_x
          
        aug_pred = self.intermediate_layer_model(aug_x, training=self.aug_grads)
        matching_pred = self.intermediate_layer_model(matching_data, training=True)
        loss += self.consistency_weight * self.compiled_loss(matching_pred, aug_pred, regularization_losses=self.losses)
          
      else: # Logit Consistency
        if self.org_matching==True:
          matching_pred = self(org_x, training=True)
        else:
          matching_pred = randaug_pred
     
        aug_pred = self(aug_x, training=self.aug_grads)
        loss += self.consistency_weight * self.compiled_loss(matching_pred, aug_pred, regularization_losses=self.losses) # todo add fine-grained loss weightings

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
