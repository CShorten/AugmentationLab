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
def create_task_groupings_header(aug_name_list):
  headings_row = []
  headings_row.append("Epoch")
  for aug_name in aug_name_list:
    headings_row.append(aug_name)
  headings_row.append("Max Test Accuracy")
  return headings_row

def get_aug_results(epoch, model, model_paths, aug_name_list,
                    x_test, y_test):
  print("evaluating...")
  new_results_row = []
  new_results_row.append(epoch)
  for model_path in model_paths:
    model.load_weights("model_paths")
    # maybe want to see the trend in training accuracy as well
    new_results_row.append(model.evaluate(x_test, y_test)[1])
  next_step_aug_idx = np.argmax(new_results_row[1:])
  new_results_row.append(aug_name_list[next_step_aug_idx])
  return new_results_row, next_step_aug_idx

def save_file(master_file, file_name):
  import csv
  with open(file_name, mode='w') as data_file:
    data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in master_file:
      data_writer.writerow(row)
