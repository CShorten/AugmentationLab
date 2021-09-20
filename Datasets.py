def get_cifar_10():
  from tensorflow import keras
  (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
  y_train = keras.utils.to_categorical(y_train, 10)
  y_test = keras.utils.to_categorical(y_test, 10)
  return x_train, y_train, x_test, y_test
