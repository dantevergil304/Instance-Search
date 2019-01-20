import numpy as np


def mean_max_similarity(query, shot_faces):
    '''
    Parameters:
    - query: [((face matrix, img query path, binary mask path), feature vector)]
    '''
    final_sim = 0
    frames_with_bb_sim = []
    for q_face in query:
        faces_sim = [(shot_face[0], cosine_similarity(
            q_face[1], shot_face[1])) for shot_face in shot_faces]
        max_sim = max(faces_sim, key=lambda x: x[1])
        final_sim += max_sim[1]
        # Each image q_face in query have a list of corresponding faces which sorted based on similarity between faces and q_face. Overall, it a matrix of faces (1)
        frames_with_bb_sim.append(faces_sim)
    return final_sim / len(query), frames_with_bb_sim


def calculate_average_faces_sim(record):
    '''
    Calculate the average similarity of all face in query w.r.t each shot.
    _____________|shotface 1|shotface 2|shotface..|shotface m|
    query_face 1 |  sim_1_1 |  sim_1_2 |    ..    |  sim_1_m |
    query_face 2 |  sim_2_1 |  sim_2_2 |    ..    |  sim_2_m |
    query_face ..|    ..    |    ..    |    ..    |    ..    |
    query_face n |  sim_n_1 |  sim_n_2 |    ..    |  sim_n_m |
    ----------------------------------------------------------
    query_face   | avg_sim_1| avg_sim_2|    ..    | avg_sim_m|

    Parameters:
    record: a list contains 3 elements:
    - shot_id
    - sim(query, shot_id): similarity between input query and current shot 
    - a matrix of shape(num_query_face, num_shot_face_detected):
        + num_query_face: #remaining faces in query after remove bad faces
        + num_shot_face_detected: #faces detected of the current shot
        + matrix[i][j]: ((frame file, bb), cosine similarity score 
    between 'query face i' and 'shot face j')

    Returns:
    - faces_data: a list of tuple (shot_id, mean_sim)
    '''
    faces_data = []
    faces_matrix = record[2]
    col = len(faces_matrix[0])
    row = len(faces_matrix)
    for i in range(col):
        data = faces_matrix[0][i][0]  # Get (frame file, bb)
        mean_sim = 0
        for j in range(row):
            mean_sim += faces_matrix[j][i][1]
        mean_sim /= row
        faces_data.append((data, mean_sim))
    return faces_data


def cosine_similarity(vector_a, vector_b):
    l2_vector_a = np.linalg.norm(vector_a) + 0.001
    l2_vector_b = np.linalg.norm(vector_b) + 0.001
    return np.dot((vector_a / l2_vector_a), (vector_b.T / l2_vector_b))
