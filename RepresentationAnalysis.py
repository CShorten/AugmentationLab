# Vector Representation Analysis

# High-Level Wrappers
def distribution_analysis(dist1, dist2):
  print(get_average_vec_distance(dist1,dist2))

def get_average_vec_distance(dist1,dist2):
  # just defaulting to l1 for now
  # dist1
  sum = 0
  for i in range(len(dist1)):
    sum += l1_vec_distance(dist1[i], dist2[i])
  return sum / len(dist1)
  
def l1_vec_distance(vec1, vec2):
  sum = 0
  for i in range(len(vec1)):
    sum += abs(vec1[i] - vec2[i])
  return sum

def l2_vec_distance(vec1, vec2):
  sum = 0
  for i in range(len(vec1)):
    sum += (vec1[i] - vec2[i])**2
  return sum

def report_vector_sparsity(vec):
  zero_sum = 0
  for i in range(len(vec)):
    if (vec[i] == 0):
      zero_sum += 1
  return zero_sum

def get_average_vec_distance(dist1,dist2):
  # just defaulting to l1 for now
  # dist1
  sum = 0
  for i in range(len(dist1)):
    sum += l1_vec_distance(dist1[i], dist2[i])
  return sum / len(dist1)

def visualize_distributions(dist1, dist2):
  # produces two continuous histograms of activation frequency
  return 0
  

