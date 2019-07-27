import glob
import os
import sys
import numpy as np


def cosine(vector_a, vector_b):
    l2_vector_a = np.linalg.norm(vector_a) + 0.001
    l2_vector_b = np.linalg.norm(vector_b) + 0.001
    return np.dot((vector_a / l2_vector_a), (vector_b.T / l2_vector_b))


def matching_audio_feat(embedding_a, embedding_b):
    sec_a = embedding_a.shape[0]
    sec_b = embedding_b.shape[0]

    embedding_b = np.concatenate((np.zeros((sec_a-1, 128)), embedding_b, np.zeros((sec_a-1, 128))))


    all_sims = []
    for i in range(0, sec_a-1 + sec_b):
        total_sim = 0
        total_embeddings = sec_a
        for j in range(0, sec_a):
            sim = cosine(embedding_a[j], embedding_b[i+j])
            total_sim += sim
            if sim == 0:
                total_embeddings -= 1

        all_sims.append(total_sim/total_embeddings)

    return max(all_sims)

def matching(query_embedding, test_embedding):
    num_q = len(query_embedding)
    total_sim = 0
    for emb in query_embedding:
        total_sim += matching_audio_feat(emb, test_embedding)

    return total_sim / num_q

    
if __name__ == '__main__':
    # e.g.: clip7 for shouting, similar for different actions 
    action_clip = sys.argv[1]

    query_embedding = []
    for np_feat in glob.glob(f'feature/{action_clip}*.npy'):
        query_embedding.append(np.load(np_feat))

    result = []
    for test_path in glob.glob('feature/*npy'):
        test_embedding = np.load(test_path)

        result.append((os.path.basename(test_path), matching(query_embedding, test_embedding)))

    result = sorted(result, key=lambda x: x[1])

    for r in result:
        print(r)
         
