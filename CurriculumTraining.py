from Checkpoints import create_file_header, get_model_results, save_file # maybe create a utils file

def static_switching(model, static_curriculum, static_curriculum_names,
                     aug_list, aug_names,
                     x_train, y_train, x_test, y_test,
                     save_file_name):
  master_file = []
  master_file.append(create_file_header(aug_names))
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
                                           str((i+1)*(j+1)*10),
                                           training_aug_name, aug_list,
                                           x_train, y_train, x_test, y_test))
  # add saving the model
  save_file(master_file, save_file_name)
        
