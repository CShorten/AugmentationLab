from Checkpoints import create_task_groupings_header, get_aug_results, save_file
'''
The idea of Task Grouping is to use a lookahead comparison of the parameters updated with another task.
Start with ø[t] and train it to candidate ø'[t+1] with each augmentation.
Select the ø'[t+1] for ø[t+1] with the highest original test accuracy.
'''
def Step_Lookahead(model, epochs,
                   aug_list, aug_name_list,
                   x_train, y_train, x_test, y_test,
                   save_file_name):
  master_file = []
  master_file.append(create_task_groupings_header(aug_name_list))
  winning_aug_idx = 0
  for i in range(epochs):
    training_aug = aug_list[winning_aug_idx]
    training_aug_name = aug_name_list[winning_aug_idx]
    print("===== Augmentation: " + training_aug_name + " =====")
    model.save_weights("previous-step-weights.h5")
    model_paths = []
    for j in range(len(aug_list)):
      model.load_weights("previous-step-weights.h5")
      # train a candidate ø'[t+1] for each augmentation
      training_aug = aug_list[j]
      augmented_images = training_aug(images=x_train)
      model.fit(augmented_images, y_train, batch_size=256, epochs=1)
      model_save_path = aug_name_list[j] + "-candidate.h5"
      model.save_weights(model_save_path)
      model_paths.append(model_save_path)
    inner_epoch_results, winning_aug_idx = get_aug_results(i, model,
                                                         model_paths, aug_name_list,
                                                         x_test, y_test)
    model.load_weights(model_paths[winning_aug_idx])
    
  
  save_file(master_file, save_file_name)
