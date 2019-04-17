import numpy as np
import pickle
import os
import shutil
import cv2


def mean_max_similarity(query, shot_faces):
    '''
    Parameters:
    - query: [(face matrix, feature vector)]
    '''
    final_sim = 0
    frames_with_bb_sim = []
    for q_face in query:
        faces_sim = [(shot_face[0], cosine_similarity(
            q_face[1], shot_face[1])) for shot_face in shot_faces]
        faces_sim = sorted(faces_sim, key=lambda x: x[1], reverse=True)
        final_sim += faces_sim[0][1]
        # Each image q_face in query have a list of corresponding faces which sorted based on similarity between faces and q_face. Overall, it a matrix of faces (1)
        frames_with_bb_sim.append(faces_sim)
    return final_sim / len(query), frames_with_bb_sim


def max_max_similarity(query, shot_faces):
    '''
    Parameters:
    - query: [(face matrix, feature vector)]
    '''
    final_sim = 0
    frames_with_bb_sim = []
    for q_face in query:
        faces_sim = [(shot_face[0], cosine_similarity(
            q_face[1], shot_face[1])) for shot_face in shot_faces]
        faces_sim = sorted(faces_sim, key=lambda x: x[1], reverse=True)
        final_sim = max(faces_sim[0][1], final_sim)
        # Each image q_face in query have a list of corresponding faces which sorted based on similarity between faces and q_face. Overall, it a matrix of faces (1)
        frames_with_bb_sim.append(faces_sim)
    return final_sim, frames_with_bb_sim


def max_mean_similarity(query, shot_faces):
    final_sim = 0
    frames_with_bb_sim = []
    n = len(query)
    total_sim_per_faces = [0] * len(shot_faces)
    for q_face in query:
        faces_sim = [(shot_face[0], cosine_similarity(
            q_face[1], shot_face[1])) for shot_face in shot_faces]
        total_sim_per_faces = [total + sim[1]
                               for total, sim in zip(total_sim_per_faces, faces_sim)]
        frames_with_bb_sim.append(faces_sim)
    return max([sim / n for sim in total_sim_per_faces]), frames_with_bb_sim


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
    - faces_data: a list of tuple ((frame file, bb), mean_sim)
    '''
    faces_data = []
    faces_matrix = record[2]
    for idx, _ in enumerate(faces_matrix):
        faces_matrix[idx] = sorted(
            faces_matrix[idx], key=lambda x: (x[0][0], x[0][1]))

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


def write_result_to_file(query_id, result, file_path):
    with open(file_path, 'w') as f:
        for i, record in enumerate(result):
            f.write(str(query_id) + ' Q0 ' + record[0] + ' ' + str(
                i + 1) + ' ' + str(record[1][0][0]) + ' STANDARD\n')


def write_result(query_id, result, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(result, f)


def create_stage_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    print("Make folder ", path)


def adjust_size_different_images(ImgList, frame_width, desire_face_width):
    MAX_HEIGHT = 0
    result_imgs = []

    frame_width = int(frame_width)
    desire_face_width = int(desire_face_width)
    for img in ImgList:
        if img is not None:
            height, width, _ = img.shape
            new_height = int(desire_face_width * height / width)
            if new_height > MAX_HEIGHT:
                MAX_HEIGHT = new_height

    for img in ImgList:
        if img is not None:
            height, width, _ = img.shape
            new_height = int(desire_face_width * height / width)

            resized_img = cv2.resize(img, (desire_face_width, new_height))
            x = np.vstack(
                tuple([resized_img, np.zeros((MAX_HEIGHT - new_height, desire_face_width, 3), dtype=np.uint8)]))
            x = np.hstack(
                tuple([x, np.zeros((MAX_HEIGHT, frame_width - desire_face_width, 3), dtype=np.uint8)]))
        else:
            x = np.zeros((MAX_HEIGHT, frame_width, 3))

        result_imgs.append(x)

    return result_imgs


def create_image_label(label, size):
    img = np.zeros(size, dtype=np.uint8)
    img.fill(255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    color = (0, 0, 0)
    line = 2
    text_size = cv2.getTextSize(label, font, font_scale, line)[0]
    textX = (img.shape[1] - text_size[0]) / 2
    textY = (img.shape[0] + text_size[1]) / 2
    cv2.putText(img, label, (int(textX), int(textY)),
                font, font_scale, color, line)
    return img


if __name__ == '__main__':
    result = [['shot239_123', 2], ['shot239_135', 3]]
    write_result_to_file(1, result, 'test.txt')
