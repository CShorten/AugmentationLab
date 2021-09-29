import numpy as np
# Take the master file as an argument and update it with the latest results
def create_file_header(eval_aug_list):
  headings_row = []
  headings_row.append("Epoch")
  headings_row.append("Training Aug")
  headings_row.append("Original Train Acc.")
  headings_row.append("Original Test Acc.")
  for aug in eval_aug_list:
    headings_row.append(aug)
  return headings_row

  
# add aug training strategy
def get_model_results(model, epoch, training_aug, eval_aug_list,
                      x_train, y_train, x_test, y_test):
  print("evaluating...")
  new_results_row = []
  new_results_row.append(epoch)
  new_results_row.append(training_aug)
  new_results_row.append(model.evaluate(x_train, y_train)[1])
  new_results_row.append(model.evaluate(x_test, y_test)[1])
  
  for aug in eval_aug_list:
    aug_test = aug(images=x_test)
    new_results_row.append(model.evaluate(aug_test, y_test)[1])
  
  return new_results_row

# reporting for Task Groupings
def create_lookahead_header(aug_name_list):
  headings_row = []
  headings_row.append("Epoch")
  for aug_name in aug_name_list:
    headings_row.append(aug_name)
  headings_row.append("Max Test Accuracy")
  return headings_row

def get_lookahead_results(epoch, model, model_paths, aug_name_list,
                    x_test, y_test):
  print("evaluating...")
  new_results_row = []
  new_results_row.append(epoch)
  for model_path in model_paths:
    model.load_weights(model_path)
    # maybe want to see the trend in training accuracy as well
    new_results_row.append(model.evaluate(x_test, y_test)[1])
  winning_aug_idx = np.argmax(new_results_row[1:])
  new_results_row.append(aug_name_list[winning_aug_idx])
  return new_results_row, winning_aug_idx

# Need to reorganize this so the file is created from the raw matrix scores
def get_groupings(model, model_init_path, performance_save_path, difference_save_path,
                         training_augs, aug_name_list,
                         x_train, y_train, x_test, y_test):
  performance_file = []
  difference_file = []
  
  headings_row = []
  headings_row.append(" ") # Offset for the matrix visualization
  for aug_name in aug_name_list:
    headings_row.append(aug_name)
  performance_file.append(headings_row)
  difference_file.append(headings_row)
  
  performance_matrix = []
  difference_matrix = []
  
  original_accuracies = []
  model.load_weights(model_init_path)
  for aug in training_augs:
    org_test_score = aug(images=x_test)
    original_accuracies.append(model.evaluate(org_test_score, y_test)[1])
  
  for i, aug in enumerate(training_augs):
    new_performance_file_row = []
    new_difference_file_row = []
    new_performance_matrix_row = []
    new_difference_matrix_row = []
    model.load_weights(model_init_path)
    augmented_images = aug(images=x_train)
    model.fit(augmented_images, y_train, batch_size=256, epochs=1)
    
    new_performance_file_row.append(aug_name_list[i])
    new_difference_file_row.append(aug_name_list[i]) # offset for visualization
    for i, test_aug in enumerate(training_augs):
      aug_test = test_aug(images=x_test)
      result = model.evaluate(aug_test, y_test)[1]
      difference = result - original_accuracies[i]
      new_performance_file_row.append(result)
      new_performance_matrix_row.append(result)
      new_difference_file_row.append(difference)
      new_difference_matrix_row.append(difference)
      
    performance_file.append(new_performance_file_row)
    performance_matrix.append(new_performance_matrix_row)
    difference_file.append(new_difference_file_row)
    difference_matrix.append(new_difference_matrix_row)
  
  save_file(difference_file, difference_save_path)
  save_file(performance_file, performance_save_path)
  return performance_matrix, difference_matrix

def get_groupings_with_subsets(model, model_init_path,
                               performance_save_path, difference_save_path,
                               training_augs, aug_name_list,
                               x_train, y_train, x_test, y_test):
  train_set_indexing = {
    0: (x_train[0:5_000], y_train[0:5_000]),
    1: (x_train[5_001:10_000], y_train[5_001:10_000]),
    2: (x_train[10_001:15_000], y_train[10_001:15_000]),
    3: (x_train[15_001:20_000], y_train[15_001:20_000]),
    4: (x_train[20_001:25_000], y_train[20_001:25_000]),
    5: (x_train[25_001:30_000], y_train[25_001:30_000]),
    6: (x_train[30_001:35_000], y_train[30_001:35_000]),
    7: (x_train[35_001:40_000], y_train[35_001:40_000]),
    8: (x_train[40_001:45_000], y_train[40_001:45_000]),
    9: (x_train[45_001:50_000], y_train[45_001:50_000])
  }
  test_set_indexing = {
    0: x_test[0:1_000], y_test[0:1_000],
    1: x_test[1_001:2_000], y_test[1_001:2_000]),
    2: x_test[2_001:3_000], y_test[2_001:3_000]),
    3: x_test[3_001:4_000], y_test[3_001:4_000]),
    4: x_test[4_001:5_000], y_test[4_001:5_000]),
    5: x_test[5_001:6_000], y_test[5_001:6_000]),
    6: x_test[6_001:7_000], y_test[6_001:7_000]),
    7: x_test[7_001:8_000], y_test[7_001:8_000]),
    8: x_test[8_001:9_000], y_test[8_001:9_000]),
    9: x_test[9_001:10_000], y_test[9_001:10_000])
  }
  performance_file = []
  difference_file = []
  
  headings_row = []
  headings_row.append(" ") # Offset for the matrix visualization
  for aug_name in aug_name_list:
    headings_row.append(aug_name)
  performance_file.append(headings_row)
  difference_file.append(headings_row)
  
  performance_matrix = []
  difference_matrix = []
  
  original_accuracies = []
  model.load_weights(model_init_path)
  for i, aug in enumerate(training_augs):
    data_subset = test_set_indexing[i]
    aug_test = aug(images=data_subset[0])
    original_accuracies.append(model.evaluate(aug_test, data_subset[1])[1])
  
  for i, aug in enumerate(training_augs):
    new_performance_file_row = []
    new_difference_file_row = []
    new_performance_matrix_row = []
    new_difference_matrix_row = []
    model.load_weights(model_init_path)
    training_subset = training_set_indexing[i]
    print("===== " + str(aug_name_list[i]) + " =====")
    augmented_images = aug(images=training_subset[0])
    model.fit(augmented_images, training_subest[1], batch_size=256, epochs=1)
    
    new_performance_file_row.append(aug_name_list[i])
    new_difference_file_row.append(aug_name_list[i]) # offset for visualization
    for i, test_aug in enumerate(training_augs):
      test_subset = testing_set_indexing[i]
      aug_test = test_aug(images=test_subset[0])
      result = model.evaluate(aug_test, test_subset[1])[1]
      difference = result - original_accuracies[i]
      new_performance_file_row.append(result)
      new_performance_matrix_row.append(result)
      new_difference_file_row.append(difference)
      new_difference_matrix_row.append(difference)
      
    performance_file.append(new_performance_file_row)
    performance_matrix.append(new_performance_matrix_row)
    difference_file.append(new_difference_file_row)
    difference_matrix.append(new_difference_matrix_row)
  
  save_file(difference_file, difference_save_path)
  save_file(performance_file, performance_save_path)
  return performance_matrix, difference_matrix

def save_file(master_file, file_name):
  import csv
  with open(file_name, mode='w') as data_file:
    data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in master_file:
      data_writer.writerow(row)
