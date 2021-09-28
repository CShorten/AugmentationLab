def standard_training(model, meta_steps, 
                      training_aug_list, testing_aug_list, 
                      x_train, y_train, x_test, y_test):
  for i in range(meta_steps):
    print("===== Training Step: " + str(i*10) + " =====")
    for aug in training_aug_list:
      augmented_data = aug(images=x_train)
      model.fit(augmented_data, y_train, batch_size=256, epochs=1)
    print("===== Evaluation Step: " + str((i+1)*10) + " =====")
    for test_aug in testing_aug_list:
      augmented_test_data = test_aug(images=x_test)
      model.evaluate(augmented_test_data, y_test, batch_size=256, epochs=1)               
