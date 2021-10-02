def standard_training(model, meta_steps, inner_steps 
                      training_aug_list, testing_aug_list, testing_aug_names, 
                      x_train, y_train, x_test, y_test):
  for i in range(meta_steps):
    print("===== Training Step: " + str(i*inner_steps) + " =====")
    for j in range(inner_steps):
      for aug in training_aug_list:
        augmented_data = aug(images=x_train)
        model.fit(augmented_data, y_train, batch_size=256, epochs=1)
    print("===== Evaluation Step: " + str((i+1)*inner_steps) + " =====")
    for k, test_aug in enumerate(testing_aug_list):
      print("====== "+ str(testing_aug_names[k]) + " Test ======")
      augmented_test_data = test_aug(images=x_test)
      model.evaluate(augmented_test_data, y_test)
      
def few_shot_training(model, model_init_path,
                      outer_epochs, inner_epochs,
                      aug_list, aug_names, negative_aug, negative_aug_name,
                      x_train,y_train,x_test,y_test):
  for i in range(len(aug_list)):
    model.load_weights(model_init_path)
    train_index = set(range(len(aug_list)))
    held_out = i
    held_out_aug = aug_list[held_out]
    train_index.remove(held_out)
    print("\n")
    print("Holding out: " + str(aug_names[held_out]) + " and " + negative_aug_name)
    print("\n")
    train_index = list(train_index)
    for j in range(outer_epochs):
      print("Training...")
      for jj in range(inner_epochs):
        for k in range(len(train_index)):
          aug = aug_list[k]
          augmented_x = aug(images=x_train)
          model.fit(augmented_x, y_train, batch_size=256, epochs=1)
      print("Zero-Shot Evaluation...")
      print(aug_names[held_out])
      test_aug_1 = aug_list[held_out](images=x_test)
      model.evaluate(test_aug_1, y_test)
      print(negative_aug_name)
      test_aug_2 = negative_aug(images=x_test)
      model.evaluate(test_aug_2, y_test)
      print("Training Augs Evaluation...")
      for i in range(len(train_index)):
        print(aug_names[train_index[i]])
        aug_test = aug_list[train_index[i]]
        aug_images = aug_test(images=x_test)
        model.evaluate(aug_images, y_test)
      
      
                      
