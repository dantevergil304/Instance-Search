import os
import glob
import sys
import numpy as np


def cosine(vector_a, vector_b):
    l2_vector_a = np.linalg.norm(vector_a) + 0.001
    l2_vector_b = np.linalg.norm(vector_b) + 0.001
    return np.dot((vector_a / l2_vector_a), (vector_b.T / l2_vector_b))


if __name__ == '__main__':
    query_path = sys.argv[1] 
    query_embedding = np.load(query_path)
    for path in glob.glob('feature/*npy'):
        audio_embedding = np.load(path)
        for query_sec_embedding in query_embedding:
            for audio_sec_embedding in audio_embedding:
                sim = cosine(query_sec_embedding, audio_sec_embedding)
                
         
