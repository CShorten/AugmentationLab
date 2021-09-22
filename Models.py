import keras
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf

from tensorflow.keras.applications import ResNet152V2, ResNet50

def standard_model(x_train, input_shape=(32,32,3)):
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

def create_specialization_head(base_output_layer):
  dense_1 = layers.Dense(512, activation="relu")(base_output_layer)
  dense_2 = layers.Dense(512, activation="relu")(dense_1)
  outputs = layers.Dense(10, activation="softmax")(dense_2)
  return outputs

def ResNet152V2_base_head(x_train, num_specialized_heads, input_shape=(32,32,3)):
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
  models = []
  for i in range(num_specialized_heads):
    head = create_specialization_head(flattened)
    model = keras.Model(inputs=inputs, outputs=head)
    models.append(model)
  return models

def ResNet50_with_upsampling(x_train, input_shape=(32,32,3)):
  normalization_layer = keras.Sequential(
    [
      layers.experimental.preprocessing.Normalization(),
      layers.experimental.preprocessing.Resizing(72, 72),
    ],
    name = "no_data_augmentation",
  )
  normalization_layer.layers[0].adapt(x_train)
  inputs = layers.Input(shape=input_shape)
  normalized = normalization_layer(inputs)
  resnet_outputs = ResNet50(weights=None, include_top=False, input_shape=(72,72,3))(normalized)
  flattened = layers.Flatten()(resnet_outputs)
  dense_1 = layers.Dense(512, activation="relu")(flattened)
  dense_2 = layers.Dense(512, activation="relu")(dense_1)
  outputs = layers.Dense(10, activation="softmax")(dense_2)
  model = keras.Model(inputs=inputs, outputs=outputs)
  return model

def compile_model(model, lr=0.001):
  model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss = keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics = [
      keras.metrics.CategoricalAccuracy(name="accuracy"),
    ],
  )

def compile_models(models, lr_list):
  for i, model in enumerate(models):
    compile_model(model, lr=lr_list[i])

# Vision Transformer
# implementation from Khalid Salama, cite: https://keras.io/examples/vision/image_classification_with_vision_transformer/
def create_vit_classifier(x_train):
  import tensorflow_addons as tfa # currently installing this in the main notebook            
  input_shape = (32,32,3)
  image_size = 72
  patch_size = 6  # Size of the patches to be extract from the input images
  num_patches = (image_size // patch_size) ** 2
  projection_dim = 64
  num_heads = 4
  transformer_units = [
      projection_dim * 2,
      projection_dim,
  ]  # Size of the transformer layers
  transformer_layers = 8
  mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier
  
  
  def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

  class Patches(layers.Layer):
    def __init__(self, patch_size):
      super(Patches, self).__init__()
      self.patch_size = patch_size

    def call(self, images):
      batch_size = tf.shape(images)[0]
      patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
      )
      patch_dims = patches.shape[-1]
      patches = tf.reshape(patches, [batch_size, -1, patch_dims])
      return patches   
      
    def get_config(self):
     config = super().get_config().copy()
     config.update({
          'patch_size': self.patch_size,
     })
     return config
    
                   
  class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
          super(PatchEncoder, self).__init__()
          self.num_patches = num_patches
          self.projection = layers.Dense(units=projection_dim)
          self.position_embedding = layers.Embedding(
              input_dim=num_patches, output_dim=projection_dim
          )

    def call(self, patch):
          positions = tf.range(start=0, limit=self.num_patches, delta=1)
          encoded = self.projection(patch) + self.position_embedding(positions)
          return encoded
    
    def get_config(self):
      config = super().get_config().copy()
      config.update({
          'num_patches': self.num_patches,
          'projection': self.projection,
          'position_embedding': self.position_embedding,
      })
      return config
    
  inputs = layers.Input(shape=input_shape)
                   
  data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
    ],
    name="data_augmentation",
  )
  # Compute the mean and the variance of the training data for normalization.
  data_augmentation.layers[0].adapt(x_train)
                   
  # Augment data.
  augmented = data_augmentation(inputs)
  # Create patches.
  patches = Patches(patch_size)(augmented)
  # Encode patches.
  encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

  # Create multiple layers of the Transformer block.
  for _ in range(transformer_layers):
      # Layer normalization 1.
      x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
      # Create a multi-head attention layer.
      attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
      )(x1, x1)
      # Skip connection 1.
      x2 = layers.Add()([attention_output, encoded_patches])
      # Layer normalization 2.
      x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
      # MLP.
      x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
      # Skip connection 2.
      encoded_patches = layers.Add()([x3, x2])

  # Create a [batch_size, projection_dim] tensor.
  representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
  representation = layers.Flatten()(representation)
  representation = layers.Dropout(0.5)(representation)
  # Add MLP.
  features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
  # Classify outputs.
  logits = layers.Dense(10, activation="softmax")(features)
  # Create the Keras model.
  model = keras.Model(inputs=inputs, outputs=logits)
  return model

def perceiver(x_train):
  num_classes=10
  dropout_rate = 0.2
  image_size = 64  # We'll resize input images to this size.
  patch_size = 2  # Size of the patches to be extract from the input images.
  num_patches = (image_size // patch_size) ** 2  # Size of the data array.
  latent_dim = 256  # Size of the latent array.
  projection_dim = 256  # Embedding size of each element in the data and latent arrays.
  num_heads = 8  # Number of Transformer heads.
  ffn_units = [
    projection_dim,
    projection_dim,
  ]  # Size of the Transformer Feedforward network.
  num_transformer_blocks = 4
  num_iterations = 2  # Repetitions of the cross-attention and Transformer modules.
  classifier_units = [
    projection_dim,
    num_classes,
  ]  # Size of the Feedforward network of the final classifier.
  def create_ffn(hidden_units, dropout_rate):
    ffn_layers = []
    for units in hidden_units[:-1]:
        ffn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    ffn_layers.append(layers.Dense(units=hidden_units[-1]))
    ffn_layers.append(layers.Dropout(dropout_rate))

    ffn = keras.Sequential(ffn_layers)
    return ffn
  
  class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
      
  class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded
   
  def create_cross_attention_module(
    latent_dim, data_dim, projection_dim, ffn_units, dropout_rate):
    inputs = {
        # Recieve the latent array as an input of shape [1, latent_dim, projection_dim].
        "latent_array": layers.Input(shape=(latent_dim, projection_dim)),
        # Recieve the data_array (encoded image) as an input of shape [batch_size, data_dim, projection_dim].
        "data_array": layers.Input(shape=(data_dim, projection_dim)),
    }

    # Apply layer norm to the inputs
    latent_array = layers.LayerNormalization(epsilon=1e-6)(inputs["latent_array"])
    data_array = layers.LayerNormalization(epsilon=1e-6)(inputs["data_array"])

    # Create query tensor: [1, latent_dim, projection_dim].
    query = layers.Dense(units=projection_dim)(latent_array)
    # Create key tensor: [batch_size, data_dim, projection_dim].
    key = layers.Dense(units=projection_dim)(data_array)
    # Create value tensor: [batch_size, data_dim, projection_dim].
    value = layers.Dense(units=projection_dim)(data_array)

    # Generate cross-attention outputs: [batch_size, latent_dim, projection_dim].
    attention_output = layers.Attention(use_scale=True, dropout=0.1)(
        [query, key, value], return_attention_scores=False
    )
    # Skip connection 1.
    attention_output = layers.Add()([attention_output, latent_array])

    # Apply layer norm.
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)
    # Apply Feedforward network.
    ffn = create_ffn(hidden_units=ffn_units, dropout_rate=dropout_rate)
    outputs = ffn(attention_output)
    # Skip connection 2.
    outputs = layers.Add()([outputs, attention_output])

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
  
  def create_transformer_module(
    latent_dim,
    projection_dim,
    num_heads,
    num_transformer_blocks,
    ffn_units,
    dropout_rate,
):

    # input_shape: [1, latent_dim, projection_dim]
    inputs = layers.Input(shape=(latent_dim, projection_dim))

    x0 = inputs
    # Create multiple layers of the Transformer block.
    for _ in range(num_transformer_blocks):
        # Apply layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(x0)
        # Create a multi-head self-attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, x0])
        # Apply layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # Apply Feedforward network.
        ffn = create_ffn(hidden_units=ffn_units, dropout_rate=dropout_rate)
        x3 = ffn(x3)
        # Skip connection 2.
        x0 = layers.Add()([x3, x2])

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=x0)
    return model
  
  data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
  )
  # Compute the mean and the variance of the training data for normalization.
  data_augmentation.layers[0].adapt(x_train)
  
  class Perceiver(keras.Model):
    def __init__(
        self,
        patch_size,
        data_dim,
        latent_dim,
        projection_dim,
        num_heads,
        num_transformer_blocks,
        ffn_units,
        dropout_rate,
        num_iterations,
        classifier_units,
    ):
        super(Perceiver, self).__init__()

        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.ffn_units = ffn_units
        self.dropout_rate = dropout_rate
        self.num_iterations = num_iterations
        self.classifier_units = classifier_units

    def build(self, input_shape):
        # Create latent array.
        self.latent_array = self.add_weight(
            shape=(self.latent_dim, self.projection_dim),
            initializer="random_normal",
            trainable=True,
        )

        # Create patching module.
        self.patcher = Patches(self.patch_size)

        # Create patch encoder.
        self.patch_encoder = PatchEncoder(self.data_dim, self.projection_dim)

        # Create cross-attenion module.
        self.cross_attention = create_cross_attention_module(
            self.latent_dim,
            self.data_dim,
            self.projection_dim,
            self.ffn_units,
            self.dropout_rate,
        )

        # Create Transformer module.
        self.transformer = create_transformer_module(
            self.latent_dim,
            self.projection_dim,
            self.num_heads,
            self.num_transformer_blocks,
            self.ffn_units,
            self.dropout_rate,
        )

        # Create global average pooling layer.
        self.global_average_pooling = layers.GlobalAveragePooling1D()

        # Create a classification head.
        self.classification_head = create_ffn(
            hidden_units=self.classifier_units, dropout_rate=self.dropout_rate
        )

        super(Perceiver, self).build(input_shape)

    def call(self, inputs):
        # Augment data.
        augmented = data_augmentation(inputs)
        # Create patches.
        patches = self.patcher(augmented)
        # Encode patches.
        encoded_patches = self.patch_encoder(patches)
        # Prepare cross-attention inputs.
        cross_attention_inputs = {
            "latent_array": tf.expand_dims(self.latent_array, 0),
            "data_array": encoded_patches,
        }
        # Apply the cross-attention and the Transformer modules iteratively.
        for _ in range(self.num_iterations):
            # Apply cross-attention from the latent array to the data array.
            latent_array = self.cross_attention(cross_attention_inputs)
            # Apply self-attention Transformer to the latent array.
            latent_array = self.transformer(latent_array)
            # Set the latent array of the next iteration.
            cross_attention_inputs["latent_array"] = latent_array

        # Apply global average pooling to generate a [batch_size, projection_dim] repesentation tensor.
        representation = self.global_average_pooling(latent_array)
        # Generate logits.
        logits = self.classification_head(representation)
        return logits
  perceiver_classifier = Perceiver(patch_size, num_patches, latent_dim, projection_dim,
                                  num_heads, num_transformer_blocks, ffn_units, dropout_rate,
                                  num_iterations, classifier_units)
  return perceiver_classifier





















