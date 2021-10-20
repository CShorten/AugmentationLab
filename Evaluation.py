# Wrapper for Vector Rep. Analysis, Confusion Matrices, ... whatever else might be useful

# Move all of this to a CSV file

# Visualize the Evaluation with a Web Interface
from RepresentationAnalysis import distribution_analysis

def create_consistency_results_header(classification_aug_names, consistency_aug_names):
  header = []
  header.append("Epoch")
  header.append("Original Train")
  header.append("Original Test")
  
  for aug_name in classification_aug_names:
    header.append(aug_name + " Train")
    header.append(aug_name + " Test")
    
  for aug_name in consistency_aug_names:
    header.append("Original<>"+aug_name)
    
  for i in range(len(consistency_aug_names)):
    for j in range(len(consistency_aug_names)):
      if (i < j):
        header.append(consistency_aug_names[i] + "<>" + consistency_aug_names[j])
  
  return header
 
def evaluate_consistency_model(epoch, model, x_train, y_train, x_test, y_test,
                               classification_augs, classification_aug_names,
                               consistency_augs, consistency_aug_names,
                               vector_consistency=False, vector_model=None):
  results = []
  results.append(epoch)
  print("\n")
  print("===== Evaluating =====")
  print("\n")
  print("==== Original Train and Test Accuracies =====")
  org_train_acc = model.evaluate(x_train, y_train, verbose=0)[1]
  org_test_acc = model.evaluate(x_test, y_test, verbose=0)[1]
  results.append(org_train_acc)
  results.append(org_test_acc)
  print("Original Train Accuracy: " + str(org_train_acc))
  print("Original Test Accuracy: " + str(org_test_acc))
  print("\n")
  for i, aug in enumerate(classification_augs):
    print("Evaluating Accuracy with Augmentation: " + str(classification_aug_names[i]))
    train_aug_set = aug(images=x_train)
    test_aug_set = aug(images=x_test)
    aug_train_acc = model.evaluate(train_aug_set, y_train, verbose=0)[1]
    aug_test_acc = model.evaluate(test_aug_set, y_test, verbose=0)[1]
    results.append(aug_train_acc)
    results.append(aug_test_acc)
    print("Tranining Accuracy with Augmentation: " + str(classification_aug_names[i]) + " = " + str(aug_train_acc))
    print("Testing Accuracy with Augmentation: " + str(classification_aug_names[i]) + " = " + str(aug_test_acc))
    print("\n")
  print("===== Evaluating Distribution Distances =====")
  if (vector_consistency == True):
    vector_sets = []
    for aug in consistency_augs:
      aug_test = aug(images=x_test)
      vector_preds = vector_model.predict(aug_test)
      vector_sets.append(vector_preds)
    
    unaugmented_preds = vector_model.predict(x_test)
    for i, vector_set in enumerate(vector_sets):
      unaugmented_dist_to_aug = distribution_analysis(unaugmented_preds, vector_set)
      print("Distance between original and " + str(consistency_aug_names[i]) + " = " + str(unaugmented_dist_to_aug))
      results.append(unaugmented_dist_to_aug)
    
    for i, vector_set_1 in enumerate(vector_sets):
      for j, vector_set_2 in enumerate(vector_sets):
        if (i < j):
          aug_dist = distribution_analysis(vector_set_1, vector_set_2)
          print("Distance between " + str(consistency_aug_names[i]) + " and " + str(consistency_aug_names[j]) + " = " + str(aug_dist))
          results.append(aug_dist)
  else:
    logit_sets = []
    for aug in consistency_augs:
      aug_test = aug(images=x_test)
      logit_preds = model.predict(aug_test)
      logit_sets.append(logit_preds)
      
    unaugmented_preds = model.predict(x_test)
    for i, logit_set in enumerate(logit_sets):
      unaugmented_dist_to_aug = distribution_analysis(unaugmented_preds, logit_set)
      print("Distance between original and " + str(consistency_aug_names[i]) + " = " + str(unaugmented_dist_to_aug))
      results.append(unaugmented_dist_to_aug)
      
    for i, logit_set_1 in enumerate(logit_sets):
      for j, logit_set_2 in enumerate(logit_sets):
        if (i < j):
          aug_dist = distribution_analysis(logit_set_1, logit_set_2)
          print("Distance between " + str(consistency_aug_names[i]) + " and " + str(consistency_aug_names[j]) + " = " + str(aug_dist))
          results.append(aug_dist)
    print("\n")
    return results
   
  
  
