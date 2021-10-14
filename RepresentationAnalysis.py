# Vector Representation Analysis
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
  
