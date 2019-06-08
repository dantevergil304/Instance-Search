from checkGBFace import GoodFaceChecker
from natsort import natsorted

import os
import glob
import cv2
import json
import sys
import numpy as np

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    with open('../cfg/config.json', 'r') as f:
        cfg = json.load(f)

    query_folder = cfg['raw_data']['queries_folder']
    query_shot_folder = '../data/raw_data/queries/2018/shot_query_inlier_faces'
    save_folder = '../data/raw_data/queries/2018/shot_query_good_faces_only'

    with open('../cfg/search_config.json') as f:
        search_cfg = json.load(f)

    rmBF_method = search_cfg['rmBadFacesMethod']
    rmBF_landmarks_params = search_cfg['rmBadFacesLandmarkBasedParams']
    rmBF_classifier_params = search_cfg['rmBadFacesClassifierParams']

    good_face_checker = GoodFaceChecker(method=rmBF_method, checkBlur=(
        rmBF_landmarks_params['is_check_blur'] == "True"))

    classifier_type = rmBF_classifier_params['model']

    names = ['chelsea', 'darrin', 'garry', 'heather',
             'jack', 'jane', 'max', 'minty', 'mo', 'zainab']
    for name in names:
        for i in range(1, 5):
            all_faces = []
            good_faces = []
            save_folder = os.path.join(
                query_folder, f'shot_query_good_faces_only/{name}/{i}')
            if not os.path.exists(save_folder):
                os.makedirs(save_folder, exist_ok=True)

            print(os.path.join(query_shot_folder, name, str(i)))
            for face_path in natsorted(glob.glob(os.path.join(query_shot_folder, name, str(i), '*png'))):
                print(face_path)
                face_img = cv2.imread(face_path)
                file_name = os.path.basename(face_path)

                print(face_img.shape)
                all_faces.append(face_img)
                if good_face_checker.isGoodFace(face_img, classifier_type=classifier_type):
                    good_faces.append(face_img)
                    cv2.imwrite(os.path.join(save_folder, file_name), face_img)
                else:
                    good_faces.append(None)

            default_face_height = 50
            visualize_faces = None
            for face, good_face in zip(all_faces, good_faces):
                height, width = face.shape[:2]
                face_width = int(default_face_height * width / height)

                face = cv2.resize(face, (face_width, default_face_height))
                if good_face is None:
                    face = np.zeros_like(face)

                if visualize_faces is None:
                    visualize_faces = face
                else:
                    visualize_faces = np.hstack((visualize_faces, face))

            visualize_face_folder = os.path.join(
                query_folder, 'visualize_shot_query_faces', f'{name}')
            if not os.path.exists(visualize_face_folder):
                os.mkdir(visualize_face_folder)
            cv2.imwrite(os.path.join(visualize_face_folder,
                                     f'goodfacetrack_peking.{i}.png'), visualize_faces)
