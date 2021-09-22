import keras

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

      for i in range(num_rep_layers):
        org_med = self.intermediate_layer_models[i].predict(org_data)
        aug_med = self.intermediate_layer_models[i].predict(aug_data)
        loss += self.compiled_loss(org_med, aug_med, regularization_losses=self.losses)
        
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    self.compiled_metrics.update_state(y, y_pred)
    return {m.name: m.result() for m in self.metrics}
  
  def call(self, data):
    return self.model(data)
