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


def BatchGenerator(faces_path, frames_path, video_face_file, VIDEO_ID=0):
    x = []
    num_faces = 0
    shots_order_and_info = []
    num_files = 0
    # generate_batch_t = time.time()
    # batch_id = 0

    # video_face_files = natsorted(glob.glob(os.path.join(
    #     faces_path, 'video' + str(VIDEO_ID), '*pickle')))
    for file in [video_face_file]:
        num_files += 1

        # print('Processing %d shot(s)' % (num_files))

        with open(file, 'rb') as f:
            bbs = pickle.load(f)

        shot_id = os.path.basename(file).split('.')[0]

        len_bbs = len(bbs)

        for frame_id, bb in bbs:  # , score in bbs:
            # if score < 0.9:
            #     len_bbs -= 1
            #     continue
            frame_path = os.path.join(
                frames_path, 'video' + str(VIDEO_ID), shot_id, frame_id)
            frame = cv2.imread(frame_path)

            x1, y1, x2, y2 = bb

            face = frame[y1:y2, x1:x2]

            face = cv2.resize(face, (224, 224))

            face = face.astype(np.float64)

            face = utils.preprocess_input(face, version=1)

            x.append(face)
            num_faces += 1

            # if num_faces == batch_size:
            #     x = np.array(x).reshape((-1, 224, 224, 3))
            #     print('Generate batch %d elapses: %d seconds' %
            #           (batch_id, time.time() - generate_batch_t))
            #     yield x
            #     x = []
            #     batch_id += 1
            #     num_faces = 0
            #     generate_batch_t = time.time()

        if len_bbs > 0:
            shots_order_and_info.append((shot_id, len_bbs))

    # if x != []:
    #     x = np.asarray(x).reshape((-1, 224, 224, 3))
    #     print('Generate batch %d elapses: %d seconds' %
    #           (batch_id, time.time() - generate_batch_t))
    #     yield x

    # yield shots_order_and_info
    return x, shots_order_and_info


def multiprocessBatchGenerator(faces_path, frames_path, VIDEO_ID=0, shots_per_batch=200, n_jobs=20):
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

        arg = [(faces_path, frames_path, video_face_files[shot_idx], VIDEO_ID)
               for shot_idx in range(start_idx, end_idx)]
        generate_batch_t = time.time()
        with multiprocessing.get_context("spawn").Pool() as pool:
            result = pool.starmap(BatchGenerator, arg)

        x, shot_order_and_info = zip(*result)
        shots_order_and_info.extend(list(shot_order_and_info))
        x = [item for sublist in x for item in sublist]
        x = np.array(x).reshape((-1, 224, 224, 3))

        print('Generate batch %d elapses: %d seconds' %
              (batch_id, time.time() - generate_batch_t))
        yield x

    yield shots_order_and_info


def extract_feat(model, faces_path, frames_path, order_info_save_path, video_id):
    begin = time.time()

    for batch in multiprocessBatchGenerator(faces_path, frames_path, video_id):
        if not isinstance(batch, np.ndarray):
            # with open(os.path.join(order_info_save_path, 'order_and_info' + str(video_id) + '.pkl'), 'wb') as f:
            #     pickle.dump(batch, f)
            continue
        print("Batch shape: %d" % batch.shape[0])

        predict_t = time.time()

        # feat = model.predict(batch, batch_size=32)
        # yield feat

        print('Predict time: %d seconds' % (time.time() - predict_t))
        print('*' * 50)

    end = time.time()
    print('Elapsed Time: %d minutes %d seconds' %
          ((end-begin)//60, (end-begin) % 60))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
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
        extract_feat(model, faces_folder, frames_folder, '../features/order_and_info', video_id)
        # all_features = None
        # for feat in extract_feat(model, faces_folder, frames_folder, '../features/order_and_info', video_id):
        #     if all_features is None:
        #         all_features = feat
        #     else:
        #         all_features = np.concatenate((all_features, feat))

        # np.save(os.path.join(default_feature_folder, 'video' +
        #                      str(video_id) + '_feat.npy'), all_features)

    print('TOTAL ELAPSED TIME: %d minutes %d seconds' %
          ((time.time() - start_t)//60, (time.time() - start_t) % 60))
