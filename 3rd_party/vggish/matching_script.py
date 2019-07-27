import numpy as np
import os
from scipy import ndimage   
import sys  

def distance(vector_a, vector_b):
    l2_vector_a = np.linalg.norm(vector_a) + 0.001
    l2_vector_b = np.linalg.norm(vector_b) + 0.001
    return np.linalg.norm(l2_vector_a - l2_vector_b)

def cosine(vector_a, vector_b):
    l2_vector_a = np.linalg.norm(vector_a) + 0.001
    l2_vector_b = np.linalg.norm(vector_b) + 0.001
    return np.dot((vector_a / l2_vector_a), (vector_b.T / l2_vector_b))

def find_min(feature_a, feature_b):
    row_a = feature_a.shape[0]
    row_b = feature_b.shape[0]
    longer, shorter = feature_a, feature_b
    if row_a < row_b:
        longer, shorter = feature_b, feature_a
    index = 0
    max_cosine=0
    list_sum = []
    while True:
        if len(shorter) + index > len(longer):
            break
        sum_dis = 0
        for i in range(len(shorter)):
            sum_dis += cosine(shorter[i], longer[i + index])
        sum_dis = sum_dis / len(shorter)
        if sum_dis >  max_cosine:
            max_cosine = sum_dis   
            list_sum.append(sum_dis)
        index += 1
            
    return max_cosine, list_sum

if __name__ == '__main__':
    files = os.listdir("feature")
    query = sys.argv[1]
    features = []
    for f in files:
        feature = np.load(os.path.join("feature", f))
        features.append([f, feature])
    ranked_list = []
    for i in range(0, len(features)):
        value, list_sum = find_min(np.load(query), features[i][1])
        ranked_list.append((features[i][0], value, list_sum))
    ranked_list = sorted(ranked_list, key=lambda x: x[1], reverse=True)
    for shot in ranked_list:
        print("Compare between %s and %s" % (query, shot[0]))
        print(shot)
