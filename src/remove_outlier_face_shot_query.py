from feature_extraction import extract_feature_from_face
from natsort import natsorted
from keras_vggface.vggface import VGGFace
from keras.engine import Model
from util import cosine_similarity

import numpy as np
import cv2
import os
import glob
import json
import sys


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    with open('../cfg/config.json', 'r') as f:
        cfg = json.load(f)

    with open('../cfg/search_config.json', 'r') as f:
        search_cfg = json.load(f)

    query_folder = cfg['raw_data']['queries_folder']
    query_shot_folder = cfg['raw_data']['query_shot_folder']

    vgg_model = VGGFace(input_shape=(224, 224, 3), pooling='avg')
    out = vgg_model.get_layer(search_cfg['feature_descriptor']).output
    model = Model(vgg_model.input, out)
    print("[+] Loaded VGGFace model")

    names = ['chelsea', 'darrin', 'garry', 'heather',
             'jane', 'jack', 'max', 'minty', 'mo', 'zainab']

    for name in names:
        for i in range(1, 5):
            print(f'[+] Processing {name} {i}')
            faces_track = []
            faces_feat_track = []
            with open(os.path.join(query_shot_folder, name, str(i), 'topic_face_index.txt'), 'r') as f:
                topic_face_index = int(f.read())
            for face_path in natsorted(glob.glob(os.path.join(query_shot_folder, name, str(i), '*png'))):

                face_img = cv2.imread(face_path)
                feat = extract_feature_from_face(model, face_img)

                faces_track.append(face_img)
                faces_feat_track.append(feat)

            inlier_faces_index = set()
            inlier_faces_index.add(topic_face_index)
            # Backward
            closestRelevantFaceIndex = topic_face_index
            for index in range(topic_face_index - 1, -1, -1):
                if cosine_similarity(faces_feat_track[index], faces_feat_track[closestRelevantFaceIndex]) >= 0.85:
                    inlier_faces_index.add(index)
                    closestRelevantFaceIndex = index

            # Forward
            closestRelevantFaceIndex = topic_face_index
            for index in range(topic_face_index + 1, len(faces_feat_track)):
                if cosine_similarity(faces_feat_track[index], faces_feat_track[closestRelevantFaceIndex]) >= 0.85:
                    inlier_faces_index.add(index)
                    closestRelevantFaceIndex = index

            # save image
            save_path = os.path.join(
                '../data/raw_data/queries/2018/shot_query_inlier_faces', name, str(i))
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            for index, face in enumerate(faces_track):
                if index in inlier_faces_index:
                    cv2.imwrite(os.path.join(
                        save_path, f'{name}.{index}.face.png'), face)

            default_face_height = 50
            visualize_faces = None
            for index, face in enumerate(faces_track):
                if index not in inlier_faces_index:
                    continue
                height, width = face.shape[:2]
                face_width = int(default_face_height * width / height)

                face = cv2.resize(face, (face_width, default_face_height))

                if visualize_faces is None:
                    visualize_faces = face
                else:
                    visualize_faces = np.hstack((visualize_faces, face))
            visualize_face_folder = os.path.join(
                query_folder, 'visualize_shot_query_faces', f'{name}')
            if not os.path.exists(visualize_face_folder):
                os.mkdir(visualize_face_folder)
            cv2.imwrite(os.path.join(visualize_face_folder,
                                     f'inlierfacetrack.{i}.png'), visualize_faces)
