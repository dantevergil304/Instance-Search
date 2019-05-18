from natsort import natsorted

import numpy as np
import cv2
import imp
import torch
import glob
import pickle
import os
import time
import sys
import json


def extendBB(org_img_size, left, top, right, bottom, ratio=0.3):
    # Params:
    # - org_img_size: a tuple of (height, width)
    width = right - left
    height = bottom - top

    new_width = width * (1 + ratio)
    new_height = height * (1 + ratio)

    center_x = (left + right) / 2
    center_y = (top + bottom) / 2

    return max(0, int(center_x - new_width/2)), max(0, int(center_y - new_height/2)), min(org_img_size[1], int(center_x + new_width/2)), min(org_img_size[1], int(center_y + new_height/2))


def BatchGeneratorVGGFace2(faces_path, frames_path, VIDEO_ID=0, batch_size=30):
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

            x1, y1, x2, y2 = extendBB(
                frame.shape[:2], bb[0], bb[1], bb[2], bb[3])

            face = frame[y1:y2, x1:x2]

            face = cv2.resize(face, (224, 224))

            face = face.astype(np.float64)

            mean = [91.4953, 103.8827, 131.0912]

            face = face - mean

            x.append(face)
            info.append(shot_id)
            num_faces += 1

            if num_faces == batch_size:
                x = np.array(x)
                print('Generate batch %d elapses: %d seconds' %
                      (batch_id, time.time() - generate_batch_t))
                yield x, info
                x = []
                info = []
                batch_id += 1
                num_faces = 0
                generate_batch_t = time.time()

    if x != []:
        x = np.array(x)
        print('Generate batch %d elapses: %d seconds' %
              (batch_id, time.time() - generate_batch_t))
        yield x, info


def extractFeatVGGFace2(model, faces_path, frames_path, video_id):
    begin = time.time()

    for batch, info in BatchGeneratorVGGFace2(faces_path, frames_path, video_id):
        print("Batch shape: %d" % batch.shape[0])

        # Preprocess Input
        batch = torch.from_numpy(batch)
        batch = batch.cuda()
        batch = batch.permute(0, 3, 1, 2)
        batch = batch.type('torch.cuda.FloatTensor')

        # Extract time
        predict_t = time.time()
        feat = model(batch)
        print('Extract time: %d seconds' % (time.time() - predict_t))
        print('*' * 50)

        # Preprocess output
        feat = feat.squeeze()
        feat = feat.unsqueeze(0)
        yield feat.data.cpu().numpy(), info

    end = time.time()
    print('Elapsed Time: %d minutes %d seconds' %
          ((end-begin)//60, (end-begin) % 60))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

    MainModel = imp.load_source(
        'MainModel', '../3rd_party/senet50_256_pytorch/senet50_256_pytorch.py')
    model = torch.load(
        '../3rd_party/senet50_256_pytorch/senet50_256_pytorch.pth')
    model = model.cuda()

    # face = cv2.resize(face, (224, 224))
    # face_tensor = torch.from_numpy(face)
    # face_tensor = face_tensor.unsqueeze(0)
    # face_tensor = face_tensor.permute(0, 3, 1, 2)
    # face_tensor = face_tensor.type('torch.FloatTensor')
    # print(face_tensor.shape)

    # out = model(face_tensor)
    # print(out)

    with open("../cfg/config.json", "r") as f:
        cfg = json.load(f)
    with open('../cfg/search_config.json', 'r') as f:
        search_cfg = json.load(f)
    print("[+] Loaded config file")

    default_feature_folder = os.path.abspath(
        cfg["features"]["VGG_default_features"])
    faces_folder = os.path.abspath(cfg["processed_data"]["faces_folder"])
    frames_folder = os.path.abspath(cfg["processed_data"]["frames_folder"])

    start_t = time.time()
    for video_id in range(0, 244):
        print('\nProcessing video %d' % video_id)
        video_features_dict = dict()
        for feats_batch, info in extractFeatVGGFace2(model, faces_folder, frames_folder, video_id):
            for feat, info in zip(feats_batch, info):
                if str(info) not in video_features_dict.keys():
                    video_features_dict[str(info)] = [feat]
                else:
                    video_features_dict[str(info)].append(feat)

        for key in video_features_dict.keys():
            video_features_dict[key] = np.array(video_features_dict[key])

        # Save features to disk
        with open(os.path.join(default_feature_folder, 'vggface2-SE-ResNet-50-256D', 'video' + str(video_id) + '.pkl'), 'wb') as f:
            pickle.dump(video_features_dict, f)

    print('TOTAL ELAPSED TIME: %d minutes %d seconds' %
          ((time.time() - start_t)//60, (time.time() - start_t) % 60))
