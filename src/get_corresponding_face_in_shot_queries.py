from keras_vggface.vggface import VGGFace
from face_extraction_queries import detect_face_by_path
from keras.engine import Model
from feature_extraction import extract_feature_from_face
from util import cosine_similarity
from natsort import natsorted
from keras import backend as K

import os
import glob
import cv2
import numpy as np
import sys
import subprocess


def getCorrFaceTrack(query_name):

    query_path = [q_path for q_path in natsorted(glob.glob(
        f'../data/raw_data/queries/2018/{query_name}*.src.png'))]
    mask_path = [m_path for m_path in natsorted(glob.glob(
        f'../data/raw_data/queries/2018/{query_name}*.mask.png'))]

    ret, _, _ = detect_face_by_path(query_path, mask_path)
    K.clear_session()
    faces, _, _ = zip(*ret)

    vgg_model = VGGFace(input_shape=(224, 224, 3), pooling='avg')
    fc7 = vgg_model.get_layer('fc7').output
    model = Model(vgg_model.input, fc7)

    face_query_feats = []
    for face in faces:
        feat = extract_feature_from_face(model, face)
        face_query_feats.append(feat.squeeze())
    face_query_avg_feat = np.average(face_query_feats, axis=0)

    corresponding_facetrack_dirs = []
    for shot_facetrack_dir in glob.glob(os.path.join(f'../data/raw_data/queries/2018/tv18.person.example.shots/{query_name}', '*_facetrack')):
        for identifier_dir in glob.glob(os.path.join(shot_facetrack_dir, '*')):
            identifier_face_feats = []
            for face_path in glob.glob(os.path.join(identifier_dir, 'faces', '*png')):
                face_img = cv2.imread(face_path)
                feat = extract_feature_from_face(model, face_img)
                identifier_face_feats.append(feat.squeeze())
            identifier_face_avg_feat = np.average(
                identifier_face_feats, axis=0)
            if cosine_similarity(face_query_avg_feat, identifier_face_avg_feat) > 0.85:
                corresponding_facetrack_dirs.append(identifier_dir)

    K.clear_session()
    return corresponding_facetrack_dirs


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    names = ['chelsea', 'darrin', 'garry', 'heather', 'max',
             'jane', 'jack', 'minty', 'mo', 'zainab']
    root_dir = '../data/raw_data/queries/2018/shot_query_face_track_without_outlier'
    # for name in names:
    #     query_dir = os.path.join(root_dir, name)
    #     if not os.path.exists(query_dir):
    #         os.mkdir(query_dir)
    #     print(f'Process {name}...')
    #     facetrack_dirs = getCorrFaceTrack(name)
    #     for facetrack_dir in facetrack_dirs:
    #         shot_id = facetrack_dir.split('/')[-2]
    #         save_path = os.path.join(query_dir, shot_id)
    #         print(save_path)
    #         if not os.path.exists(save_path):
    #             os.mkdir(save_path)
    #         subprocess.call(['cp', '-r', facetrack_dir, save_path])

    max_num_face_per_shot = 10
    shot_query_face_dir = '../data/raw_data/queries/2018/shot_query_face'
    for query_dir in glob.glob(os.path.join(root_dir, '*')):
        query_name = os.path.basename(query_dir)

        query_name_dir = os.path.join(shot_query_face_dir, query_name)
        if not os.path.exists(os.path.join(shot_query_face_dir, query_name)):
            os.mkdir(query_name_dir)

        for shot_dir in glob.glob(os.path.join(query_dir, '*')):
            print('Shot Directory:', shot_dir)
            shot_name = os.path.basename(shot_dir)
            if not os.path.exists(os.path.join(query_name_dir, shot_name)):
                os.mkdir(os.path.join(query_name_dir, shot_name))
            img_name_list = []
            for img_path in glob.iglob(os.path.join(shot_dir, '**', 'faces', '*png'), recursive=True):
                img_name_list.append(img_path)
            img_name_list = sorted(
                img_name_list, key=lambda x: float(x.split('/')[-1].split('-')[0]))
            coef = len(img_name_list) / max_num_face_per_shot
            for i in range(max_num_face_per_shot):
                img_path = img_name_list[int(i*coef)]
                print('Image path:', img_path)
                subprocess.call(
                    ['cp', img_path, os.path.join(query_name_dir, shot_name)])
