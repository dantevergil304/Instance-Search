from natsort import natsorted
import os
import glob
import json
import cv2

if __name__ == '__main__':
    with open('../cfg/config.json', 'r') as f:
        cfg = json.load(f)

    with open('../cfg/search_config.json', 'r') as f:
        search_cfg = json.load(f)

    query_folder = cfg['raw_data']['queries_folder']
    inlier_faces_folder = '../data/raw_data/queries/2018/shot_query_inlier_faces'

    names = ['chelsea', 'darrin', 'garry', 'heather',
             'jane', 'jack', 'max', 'minty', 'mo', 'zainab']

    num_sampling = 10
    for name in names:
        for i in range(1, 5):
            print(f'[+] Processing {name} {i}')
            faces = []
            files_names = []
            for face_path in natsorted(glob.glob(os.path.join(inlier_faces_folder, name, str(i), '*png'))):
                face_img = cv2.imread(face_path)
                faces.append(face_img)
                files_names.append(os.path.basename(face_path))

            step = int(len(faces) / num_sampling)
            if len(faces) < num_sampling:
                step = 1

            save_path = os.path.join(
                '../data/raw_data/queries/2018/sampling_shot_query_faces', name, str(i))
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            for i in range(0, len(faces), step):
                cv2.imwrite(os.path.join(save_path, files_names[i]), faces[i])
