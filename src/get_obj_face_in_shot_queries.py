from keras_vggface.vggface import VGGFace
from face_extraction_queries import detect_face_by_path
from keras.engine import Model
from feature_extraction import extract_feature_from_face
from face_extraction import extract_faces_from_image
from util import cosine_similarity

import glob
import os
import cv2
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join('..', '3rd_party')))
from ServiceMTCNN import detect_face as lib

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

    sess = tf.Session()
    pnet, rnet, onet = lib.create_mtcnn(sess, None)

    vgg_model = VGGFace(input_shape=(224, 224, 3), pooling='avg')
    out = vgg_model.get_layer('fc7').output
    model = Model(vgg_model.input, out)

    for query_dir in glob.glob('../data/raw_data/queries/2018/shot_query_frames/5fps/*'):
        query_name = os.path.basename(query_dir)

        query_path = []
        mask_path = []
        for idx in range(1, 5):
            query_path.append(os.path.join(
                '../data/raw_data/queries/2018', f'{query_name}.{idx}.src.png'))
            mask_path.append(os.path.join(
                '../data/raw_data/queries/2018', f'{query_name}.{idx}.mask.png'))

        # detect faces in query
        print(query_path)
        query_faces, _, _ = detect_face_by_path(query_path, mask_path)

        # extract feat of faces in query

        query_faces_features = []
        for face, _, _ in query_faces:
            query_faces_features.append(extract_feature_from_face(model, face))

        # detect faces in shot query
        shot_query_faces = []
        for frame_path in glob.iglob(os.path.join(query_dir, '**', '*png'), recursive=True):
            frame = cv2.imread(frame_path)
            shot_query_faces.extend(
                extract_faces_from_image(frame, pnet, rnet, onet))

        # extract feat of faces in shot query
        shot_query_faces_features = []
        for face in shot_query_faces:
            shot_query_faces_features.append(
                extract_feature_from_face(model, face))

        save_path = os.path.join(
            '../data/raw_data/queries/2018/shot_query_faces/5fps', query_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for idx, (sqface, sqffeat) in enumerate(zip(shot_query_faces, shot_query_faces_features)):
            max_sim = 0
            for qffeat in query_faces_features:
                max_sim = max(cosine_similarity(qffeat, sqffeat), max_sim)
            if max_sim > 0.9:
                cv2.imwrite(os.path.join(
                    save_path, f'{query_name}.{idx}.png'), sqface)
