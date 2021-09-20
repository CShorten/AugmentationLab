def model_list_aug_results(model_list, model_names,
                    aug_list, aug_name_list,
                    x_train, y_train,
                    x_test, y_test):
  train_results = []
  test_results = []
  # create headers
  train_header = []
  test_header = []
  train_header.append("Model Name")
  train_header.append("Original")
  test_header.append("Model Name")
  test_header.append("Original")
  for aug_name in aug_name_list:
    train_header.append(aug_name)
    test_header.append(aug_name)
  train_results.append(train_header)
  test_results.append(test_header)
  # get results
  for i, model in enumerate(model_list):
    train_results_row = []
    test_results_row = []
    train_results_row.append(model_names[i])
    test_results_row.append(model_names[i])
    train_results_row.append(model.evaluate(x_train, y_train)[1])
    test_results_row.append(model.evaluate(x_test, y_test)[1])
    for aug in aug_list:
      aug_train = aug(images=x_train)
      train_results_row.append(model.evaluate(aug_train, y_train)[1])
      aug_test = aug(images=x_test)
      test_results_row.append(model.evaluate(aug_test, y_test)[1])
    train_results.append(train_results_row)
    test_results.append(test_results_row)
  return train_results, test_results
  
