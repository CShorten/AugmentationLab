from Checkpoints import create_file_header, create_task_groupings_header, get_model_results, get_aug_results, save_file
import numpy as np

def static_switching(model, static_curriculum, static_curriculum_names,
                     aug_list, aug_names,
                     x_train, y_train, x_test, y_test,
                     save_file_name):
  master_file = []
  master_file.append(create_file_header(aug_names))
  epoch_counter = 10
  for i in range(len(static_curriculum)):
    training_aug = static_curriculum[i]
    training_aug_name = static_curriculum_names[i]
    print("===== " + training_aug_name + " =====")
    # todo, change these to function arguments
    for j in range(5):
      for k in range(10):
        augmented_images = training_aug(images=x_train)
        model.fit(augmented_images, y_train, batch_size=256, epochs=1)
      master_file.append(get_model_results(model,
                                           epoch_counter,
                                           training_aug_name, aug_list,
                                           x_train, y_train, x_test, y_test))
      epoch_counter += 10
  # add saving the model
  save_file(master_file, save_file_name)

# refactor so eval augs and curriculum augs can be different
def AugSwitch(model, outer_epochs, inner_epochs,
              aug_list, aug_names,
              x_train, y_train, x_test, y_test,
              save_file_name):
  master_file = []
  master_file.append(create_file_header(aug_names))
  last_step_accuracy_array = [0] * len(aug_list)
  accuracy_difference_array = [0] * len(aug_list)
  epoch_counter = inner_epochs
  
  for i in range(outer_epochs):
    aug_index_key = np.argmax(accuracy_difference_array)
    training_aug = aug_list[aug_index_key]
    training_aug_name = aug_names[aug_index_key]
    print("===== Augmentation: " + training_aug_name + " =====")
    for j in range(inner_epochs):
      # todo - change to a dataloader with a sampling argument
      augmented_images = training_aug(images=x_train)
      model.fit(augmented_images, y_train, batch_size=256, epochs=1)
    inner_epoch_results = get_model_results(model,
                                            epoch_counter,
                                            training_aug_name, aug_list,
                                            x_train, y_train, x_test, y_test)
    master_file.append(inner_epoch_results)
    for k in range(len(aug_list)):
      if i == 0:
        last_step_accuracy_array[k] = inner_epoch_results[k+4]
      else:
        accuracy_difference_array[k] = abs(last_step_accuracy_array[k] - inner_epoch_results[k+4])
        last_step_accuracy_array[k] = inner_epoch_results[k+4] # 4 to offset (epoch, name, org_train, org_test)
    
    epoch_counter += inner_epochs
  
  save_file(master_file, save_file_name)

'''
The idea of Task Grouping is to use a lookahead comparison of the parameters updated with another task.
Start with ø[t] and train it to candidate ø'[t+1] with each augmentation.
Select the ø'[t+1] for ø[t+1] with the highest original test accuracy.
'''
def Task_Groupings(model, epochs,
                   aug_list, aug_names,
                   x_train, y_train, x_test, y_test,
                   save_file_name):
  master_file = []
  master_file.append(create_task_groupings_header(aug_names))
  next_step_aug_idx = 0
  for i in range(epochs):
    training_aug = aug_list[next_step_aug_idx]
    training_aug_name = aug_names[next_step_aug_idx]
    print("===== Augmentation: " + training_aug_name + " =====")
    model.save_weights("previous-step-weights.h5")
    model_paths = []
    for j in range(len(aug_list)):
      model.load_weights("previous-step-weights.h5")
      # train a candidate ø'[t+1] for each augmentation
      training_aug = aug_list[j]
      augmented_images = training_aug(images=x_train)
      model.fit(augmented_images, y_train, batch_size=256, epochs=1)
      model_save_path = aug_names[j] + "-candidate.h5"
      model.save_weights(model_save_path)
      model_paths.append(model_save_path)
    inner_epoch_results, next_step_aug_idx = get_aug_results(i, model,
                                                         model_paths, aug_name_list,
                                                         x_test, y_test)
    
  
  save_file(master_file, save_file_name)
        
