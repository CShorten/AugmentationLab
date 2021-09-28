def create_results_header(self, aug_name_list, training_strategy):
  header_list = []
  header_list.append(training_strategy)
  header_list.append("Epoch")
  header_list.append("Original Train Accuracy")
  header_list.append("Original Test Accuracy")
  for aug in aug_name_list:
    header_list.append(aug)
  header_list.append("Mean Aug Score")
  header_list.append("LOO Aug Score") # LOO = Last One Out, not including the aug that was just used for training in the average
  # might also want to see the performance on the augmented train sets
  # maybe also include the local changes between evaluations in the report file
  return header_list
 
def evalutation(self, aug_list, aug_name, AugSwitch=False, **last_step_array):
  evaluation_step_results = []
  if (AugSwitch == True):
    difference_array = [0] * len(aug_list)
    new_last_step_array = [0] * len(aug_list)
  for i in range(len(aug_list)):
    augmented_data = aug_list[i](images=self.x_test)
    evaluation_result = self.model.evaluate(augmented_data, self.y_test, verbose=1)[1] # check on verbose parameter for silent mode with this
    print("Evaluation on " + str(aug_name[i]) + ": " + str(evaluation_result))
    evaluation_step_results.append(evaluation_result)
    if (AugSwitch == True):
      difference_array[i] = abs(last_step_array[i] - evaluation_result)
      new_last_step_array[i] = evaluation_result
  if (AugSwitch == True):
    return difference_array, new_last_step_array, evaluation_step_results
  else:
    return evaluation_step_results
 
def build_aug_dicts(self, aug_list, aug_name_list):
  aug_dict = {}
  for i in range(len(aug_list)):
    aug_dict[i] = aug_list[i]
  aug_name_dict = {}
  for j in range(len(aug_name_list)):
    aug_name_dict[j] = aug_name_list[j]
  return (aug_dict, aug_name_dict)
  
def save_results(self, report_list, filename):
  with open(filename + '.csv', mode='w') as data_file:
  data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  for row in report_list:
    data_writer.writerow(row)

def static_training(self, aug, aug_name, steps, evaluation_steps, report_name):
  report = []
  report.append(self.create_results_header(aug_name, "Static"))
  for i in range(steps):
    if i % evaluation_steps == 0:
      print("===== Evaluating =====")
      report.append(self.evaluation_protocol(self.aug_list, self.aug_name_list))
    augmented_data = aug(images=self.x_train)
    print("===== " + str(i) + " =====")
    model.fit(augmented_data, self.y_train, batch_size=256, epochs=1)
  self.save_results(report, "Static-Training-"+aug_name+"-"+Steps+".csv")
  print("Finished Static Training")
    
def fixed_switching(self, aug_list, aug_name_list, outer_steps, inner_steps):
  report = []
  report.append(self.create_results_header(self.aug_list, self.aug_name_list, "Static Switching"))
  for j in range(outer_steps):
    aug = aug_list[j]
    for i in range(inner_steps):
      augmented_data = aug(images=self.x_train)
    report.append(self.evaluation(aug_list, aug_name_list)
  self.save_results(report, "FixedSwitching.csv") # extend to maybe use an acronym of the switches in the filename
  print("Finished Training")
                    
def AugSwitch(self, aug_list, aug_name_list, outer_steps, inner_steps):
  report = []
  last_step_array = [0] * len(aug_list)
  difference_array = [0] * len(aug_list)
  key = np.argmax(difference_array)
  for i in range(outer_steps):
    last_step_accuracy_array, difference_array, report_update = self.evaluation(aug_list, aug_name_list,
                                                                     AugSwitch=True, last_step_array, difference_array)
    report.append(report_update)
    key = np.argmax(difference_array)
    for j in range(inner_steps):
      augmented_data = aug(images=self.x_train)
      print("===== Inner Step: " + str(j) + " =====")
      model.fit(augmented_data, self.y_train, batch_size=256, epochs=1)
    
  self.save_results(report, "AugSwitch-"+str(outer_steps)+"-"+str(inner_steps)+".csv")
  print("Finished Training")
