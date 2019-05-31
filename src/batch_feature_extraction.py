from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.engine import Model
from natsort import natsorted

import cv2
import glob
import os
import pickle
import numpy as np
import time
import json
import sys
import multiprocessing
import math


def BatchGenerator(faces_path, frames_path, VIDEO_ID=0, batch_size=2000):
    x = []
    info = []
    num_faces = 0
    generate_batch_t = time.time()
    batch_id = 0

    video_face_files = natsorted(glob.glob(os.path.join(
        faces_path, 'video' + str(VIDEO_ID), '*pickle')))
    for file in video_face_files:
        with open(file, 'rb') as f:
            bbs = pickle.load(f)

        shot_id = os.path.basename(file).split('.')[0]

        for frame_id, bb in bbs:
            frame_path = os.path.join(
                frames_path, 'video' + str(VIDEO_ID), shot_id, frame_id)
            frame = cv2.imread(frame_path)

            x1, y1, x2, y2 = bb

            face = frame[y1:y2, x1:x2]

            face = cv2.resize(face, (224, 224))

            face = face.astype(np.float64)

            face = utils.preprocess_input(face, version=1)

            x.append(face)
            info.append(shot_id)
            num_faces += 1

            if num_faces == batch_size:
                x = np.array(x).reshape((-1, 224, 224, 3))
                print('Generate batch %d elapses: %d seconds' %
                      (batch_id, time.time() - generate_batch_t))
                yield x, info
                x = []
                info = []
                batch_id += 1
                num_faces = 0
                generate_batch_t = time.time()

    if x != []:
        x = np.asarray(x).reshape((-1, 224, 224, 3))
        print('Generate batch %d elapses: %d seconds' %
              (batch_id, time.time() - generate_batch_t))
        yield x, info


def multiprocessBatchGenerator(faces_path, frames_path, VIDEO_ID=0, shots_per_batch=200):
    video_face_files = natsorted(glob.glob(os.path.join(
        faces_path, 'video' + str(VIDEO_ID), '*pickle')))
    num_video_face_files = len(video_face_files)

    shots_order_and_info = []
    for batch_id in range(math.ceil(num_video_face_files / shots_per_batch)):
        start_idx = batch_id * shots_per_batch
        end_idx = start_idx + shots_per_batch
        if end_idx > num_video_face_files:
            end_idx = num_video_face_files
        print('Block Interval:', (start_idx, end_idx))

        generate_batch_t = time.time()
        arg = [(faces_path, frames_path, video_face_files[shot_idx], VIDEO_ID)
               for shot_idx in range(start_idx, end_idx)]
        with multiprocessing.get_context("spawn").Pool(processes=15) as pool:
            result = pool.starmap(BatchGenerator, arg)

        x, shot_order_and_info = zip(*result)
        shots_order_and_info.extend(list(shot_order_and_info))
        x = [item for sublist in x for item in sublist]
        x = np.array(x).reshape((-1, 224, 224, 3))

        print('Generate batch %d elapses: %d seconds' %
              (batch_id, time.time() - generate_batch_t))
        yield x

    yield shots_order_and_info


def extract_feat(model, faces_path, frames_path, video_id):
    begin = time.time()

    for batch, info in BatchGenerator(faces_path, frames_path, video_id):
        print("Batch shape: %d" % batch.shape[0])

        predict_t = time.time()

        feat = model.predict(batch, batch_size=20)
        print('Extract time: %d seconds' % (time.time() - predict_t))
        print('*' * 50)

        yield feat, info

    end = time.time()
    print('Elapsed Time: %d minutes %d seconds' %
          ((end-begin)//60, (end-begin) % 60))


def extract_database_faces_features(feature_extractor, frames_path, faces_path, features_path):
    start_t = time.time()
    for video_id in range(140, 244):
        print('\nProcessing video %d' % video_id)
        if os.path.exists(os.path.join(features_path, 'video' + str(video_id) + '.pkl')):
            print(f'File video{video_id}.pkl already existed!!!')
            continue
        video_features_dict = dict()
        for feats_batch, info in extract_feat(feature_extractor, faces_path, frames_path, video_id):
            for feat, info in zip(feats_batch, info):
                if info not in video_features_dict.keys():
                    video_features_dict[info] = [feat]
                else:
                    video_features_dict[info].append(feat)

        for key in video_features_dict.keys():
            video_features_dict[key] = np.array(video_features_dict[key])

        # Save features to disk
        with open(os.path.join(features_path, 'video' + str(video_id) + '.pkl'), 'wb') as f:
            pickle.dump(video_features_dict, f)

    print('TOTAL ELAPSED TIME: %d minutes %d seconds' %
          ((time.time() - start_t)//60, (time.time() - start_t) % 60))


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2'
    with open("../cfg/config.json", "r") as f:
        cfg = json.load(f)
    with open('../cfg/search_config.json', 'r') as f:
        search_cfg = json.load(f)
    print("[+] Loaded config file")

    default_feature_folder = os.path.abspath(
        cfg["features"]["VGG_default_features"])
    faces_folder = os.path.abspath(cfg["processed_data"]["faces_folder"])
    frames_folder = os.path.abspath(cfg["processed_data"]["frames_folder"])

    vgg_model = VGGFace(input_shape=(224, 224, 3), pooling='avg')
    out = vgg_model.get_layer(search_cfg['feature_descriptor']).output
    model = Model(vgg_model.input, out)
    print("[+] Loaded VGGFace model")

    start_t = time.time()
    for video_id in range(0, 244):
        print('\nProcessing video %d' % video_id)
        extract_feat(model, faces_folder, frames_folder,
                     '../features/order_and_info', video_id)
        all_features = None
        for feat in extract_feat(model, faces_folder, frames_folder, '../features/order_and_info', video_id):
            if all_features is None:
                all_features = feat
            else:
                all_features = np.concatenate((all_features, feat))

        np.save(os.path.join(default_feature_folder, 'video' +
                             str(video_id) + '_feat.npy'), all_features)

    print('TOTAL ELAPSED TIME: %d minutes %d seconds' %
          ((time.time() - start_t)//60, (time.time() - start_t) % 60))
