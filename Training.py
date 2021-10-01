def standard_training(model, meta_steps, 
                      training_aug_list, testing_aug_list, testing_aug_names, 
                      x_train, y_train, x_test, y_test):
  for i in range(meta_steps):
    print("===== Training Step: " + str(i*10) + " =====")
    for aug in training_aug_list:
      augmented_data = aug(images=x_train)
      model.fit(augmented_data, y_train, batch_size=256, epochs=1)
    print("===== Evaluation Step: " + str((i+1)*10) + " =====")
    for i, test_aug in enumerate(testing_aug_list):
      print("====== "+ str(testing_aug_names[i]) + " Test ======")
      augmented_test_data = test_aug(images=x_test)
      model.evaluate(augmented_test_data, y_test)
      
def few_shot_training(model, model_init_path,
                      outer_epochs, inner_epochs,
                      aug_list, aug_names,
                      x_train,y_train,x_test,y_test):
  for i in range(len(aug_list)):
    model.load_weights(model_init_path)
    train_index = set(range(len(aug_list)))
    held_out = i, (i+1)%len(aug_list)
    held_out_aug_1, held_out_aug_2 = aug_list[held_out[0]], aug_list[held_out[1]]
    train_index.remove(held_out[0])
    train_index.remove(held_out[1])
    print("\n")
    print("Holding out: " + str(aug_names[held_out[0]]) + " and " + str(aug_names[held_out[1]]))
    print("\n")
    train_index = list(train_index)
    for j in range(outer_epochs):
      print("Training...")
      for jj in range(inner_epochs):
        for k in range(len(train_index)):
          aug = aug_list[k]
          augmented_x = aug(images=x_train)
          model.fit(augmented_x, y_train, batch_size=256, epochs=1)
      print("Evaluating...")
      print(aug_names[held_out[0]])
      test_aug_1 = held_out_aug_1(images=x_test)
      model.evaluate(test_aug_1, y_test)
      print(aug_names[held_out[1]])
      test_aug_2 = held_out_aug_2(images=x_test)
      model.evaluate(test_aug_2)
                      
      
                      
