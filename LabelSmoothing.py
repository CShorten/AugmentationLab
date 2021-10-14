import numpy as np
def uniform_noise(labels):
  new_labels = []
  for label_array in labels:
    # compute how much to take off the array
    # e.g. 10 labels -- 9 slots for 18%/2% each
    label_array_copy = [0] * len(label_array)
    class_idx = np.argmax(label_array)
    label_array_copy[class_idx] = 0.82
    label_list = set(range(10))
    label_list.remove(class_idx)
    for idx in label_list:
      label_array_copy[idx] = 0.02
    new_labels.append(label_array_copy)
  return np.array(new_labels)

# Use with the Dataset Loader

y_train = uniform_noise(y_train) # maybe want to change name as well
