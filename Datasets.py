from tensorflow import keras
  
def get_cifar_10():
  (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
  y_train = keras.utils.to_categorical(y_train, 10)
  y_test = keras.utils.to_categorical(y_test, 10)
  return x_train, y_train, x_test, y_test

def get_cifar10_deer_vs_horses():
  (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
  # data filtering based on labeling
  filtered_x_train, filtered_y_train = [], []
  for i, y in enumerate(y_train):
    if (y == 4):
      filtered_x_train.append(x_train[i])
      filetered_y_train.append(y)
    if (y == 7):
      filtered_x_train.append(x_train[i])
      filtered_y_train.append(y)
      
  for i, y in enumerate(y_test):
    if (y == 4):
      filtered_x_train.append(x_train[i])
      filetered_y_train.append(y)
    if (y == 7):
      filtered_x_train.append(x_train[i])
      filtered_y_train.append(y)
      
  filtered_y_train = keras.utils.to_categorical(filtered_y_train, 10)
  filtered_y_test = keras.utils.to_categorical(filtered_y_test, 10)
  return filtered_x_train, filtered_y_train, filtered_x_test, filtered_y_test
