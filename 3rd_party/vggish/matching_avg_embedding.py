from natsort import natsorted

import os
import glob
import sys
import numpy as np


def distance(vector_a, vector_b):
    l2_vector_a = np.linalg.norm(vector_a) + 0.001
    l2_vector_b = np.linalg.norm(vector_b) + 0.001
    return np.linalg.norm(l2_vector_a - l2_vector_b)


def cosine(vector_a, vector_b):
    l2_vector_a = np.linalg.norm(vector_a) + 0.001
    l2_vector_b = np.linalg.norm(vector_b) + 0.001
    return np.dot((vector_a / l2_vector_a), (vector_b.T / l2_vector_b))


def avg_embedding(audio_embedding):
    return np.mean(audio_embedding, axis=0)


if __name__=='__main__':
    # e.g.: clip7 or something simlilar
    query_embedding = sys.argv[1]

    embeddings = []
    for feat_path in glob.glob(f'feature/{query_embedding}*.npy'):
        avg_embedding_query = avg_embedding(np.load(feat_path))
        embeddings.append(avg_embedding_query)

    sim_matrix = []
    for emb in embeddings:
        result = []
        for path in natsorted(glob.glob('feature/*npy')):

            audio_embedding = np.load(path)

            avg_audio_embedding = avg_embedding(audio_embedding)

            result.append(cosine(emb, avg_audio_embedding))

        sim_matrix.append(np.array(result))

    sim_vector = np.sum(np.array(sim_matrix), axis=0)


    result = list(zip(natsorted(glob.glob('feature/*npy')), sim_vector))
    result = sorted(result, key=lambda x: x[1])
    # result = sorted(result, key=lambda x: x[0])


    for rank, r in enumerate(result):
        print(f'rank {rank}:', r)
        
        
