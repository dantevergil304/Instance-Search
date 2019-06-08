from keras_vggface.vggface import VGGFace
from sklearn.cluster import KMeans
from keras.engine import Model
from feature_extraction import extract_feature_from_face
from util import cosine_similarity
from face_extraction_queries import detect_face_by_path
from natsort import natsorted
from keras import backend as K

import os
import subprocess
import glob
import cv2
import numpy as np
import sys

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

    vgg_model = VGGFace(input_shape=(224, 224, 3), pooling='avg')
    fc7 = vgg_model.get_layer('fc7').output
    model = Model(vgg_model.input, fc7)

    root_dir = '../data/raw_data/queries/2018/shot_query_face_track'
    save_dir = '../data/raw_data/queries/2018/shot_query_face_track_without_outlier'
    for query_dir in glob.glob(os.path.join(root_dir, '*')):
        subprocess.call(['cp', '-r', query_dir, save_dir])

    for query_dir in glob.glob(os.path.join(save_dir, '*')):
        query_name = os.path.basename(query_dir)
        query_path = [q_path for q_path in natsorted(glob.glob(
            f'../data/raw_data/queries/2018/{query_name}*.src.png'))]
        mask_path = [m_path for m_path in natsorted(glob.glob(
            f'../data/raw_data/queries/2018/{query_name}*.mask.png'))]

        ret, _, _ = detect_face_by_path(query_path, mask_path)
        faces, _, _ = zip(*ret)

        face_query_feats = []
        for face in faces:
            feat = extract_feature_from_face(model, face)
            face_query_feats.append(feat.squeeze())
        face_query_avg_feat = np.average(face_query_feats, axis=0)

        query_save_dir = os.path.join(save_dir, os.path.basename(root_dir))
        for shot_dir in glob.glob(os.path.join(query_dir, '*')):
            face_feats = []
            face_paths = []
            for face_path in glob.iglob(os.path.join(shot_dir, '**', '*png'), recursive=True):
                face_img = cv2.imread(face_path)
                feat = extract_feature_from_face(model, face_img)
                face_feats.append(feat.squeeze())
                face_paths.append(face_path)

            face_feats = np.array(face_feats)

            kmeans = KMeans(n_clusters=2, random_state=42).fit(face_feats)

            outlier_cluster_idx = []
            for idx, cluster in enumerate(kmeans.cluster_centers_):
                if cosine_similarity(face_query_avg_feat, cluster) < 0.8:
                    outlier_cluster_idx.append(idx)

            for face_path, cluster_idx in zip(face_paths, kmeans.labels_):
                if cluster_idx in outlier_cluster_idx:
                    print('Removing', face_path)
                    subprocess.call(['rm', face_path])
