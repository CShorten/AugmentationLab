def model_list_aug_results(model_list, model_names,
                    aug_list, aug_name_list,
                    x_train, y_train,
                    x_test, y_test):
  train_results = []
  test_results = []
  # create headers
  train_results.append("Model Name")
  test_results.append("Model Name")
  train_results.append("Original")
  test_results.append("Original")
  for aug_name in aug_name_list:
    train_results.append(aug_name)
    test_results.append(aug_name)
  # get results
  for i, model in enumerate(model_list):
    train_results.append(model_names[i])
    test_results.append(model_names[i])
    train_results.append(model.evaluate(x_train, y_train))[1]
    test_results.append(model.evaluate(x_test, y_test))[1]
    for aug in aug_list:
      aug_train = aug(images=x_train)
      train_results.append(model.evaluate(aug_train, y_train))[1]
      aug_test = aug(images=x_test)
      test_results.append(model.evaluate(aug_test, y_test))[1]
  return train_results, test_results
  
