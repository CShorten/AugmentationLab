import numpy as np

def fixed_sampling(imbalance_pct, minority_label, xs, ys):
  ret_xs = []
  ret_ys = []
  
  minority_count = len(xs) * imbalance_pct
  counter = 0
  
  for i, y in enumerate(ys):
      if counter > minority_count:
        break
      if (np.argmax(y) == minority_label):
        ret_xs.append(xs[i])
        ret_ys.append(y)
        counter += 1
  
  return ret_xs, ret_ys

# extension of above to not sample left to right for the imbalance pct
def random_sampling():
  return 0

# Automatic Domain Randomizationw rapper of random_sampling
def automatic_imbalance_randomization():
  # calls the above functions for k steps
  # might but this in the a trianing loop object
  return 0
