import numpy as np
import base64
import requests
import json
import os
import glob
import pickle
import time


def extract_feature_from_face_ArcFace(frame_path, face_bounding_box):
    '''
    Params:
    frame_path: path to the frame file
    face_bounding_box: bbox of a face need extracting feature (left, top, right, bottom)

    Return:

    '''
    url = 'http://192.168.28.40/face_recognition/get_feature/post/'

    # image_path = "test.jpg"
    image = open(frame_path, 'rb')
    image_read = image.read()
    encoded = base64.encodestring(image_read)
    encoded_string = encoded.decode('utf-8')

    data = {"data": {"username": "mmlab", "password": "mmlab", "mode": "remote", "modeltype": "ArcFace", "backbone": "default", "bboxes": [
        list(face_bounding_box)], "landmarks": [{"have_landmark": "False"}], "image": encoded_string}}

    headers = {'Content-type': 'application/json'}
    data_json = json.dumps(data)

    while True:
        response = requests.post(url, data=data_json, headers=headers)
        response_json = response.json()
        response_code = response_json['code']
        if response_code == '200':
            break

    for feature in zip(response_json['data']['features']):
        decoded_string = base64.b64decode(feature[0].encode())
        feat = np.frombuffer(decoded_string, dtype=np.float32, count=-1)

    # print(response_json['data']['process_time'])

    return feat


if __name__ == '__main__':
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
    for video_id in range(92, 170):
        print('\nProcessing video %d' % video_id)
        video_features_dict = dict()

        for idx, shot_faces_path in enumerate(glob.glob(os.path.join(faces_folder, f'video{video_id}', '*pickle'))):
            shotId = os.path.splitext(os.path.basename(shot_faces_path))[0]
            # if shotId != 'shot1_284':
            # continue
            print(f'[+] shot file {idx}:', shotId)
            try:
                with open(shot_faces_path, 'rb') as f:
                    shot_faces = pickle.load(f)
            except EOFError:
                print('Warning: %s causing EOF error!' % shotId)
                continue

            all_feats = []
            for frame_id, bbox in shot_faces:
                frame_path = os.path.join(
                    frames_folder, f'video{video_id}', shotId, frame_id)
                feat = extract_feature_from_face_ArcFace(frame_path, bbox)
                all_feats.append([feat.tolist()])
            video_features_dict[shotId] = np.array(all_feats, dtype=np.float32)

        # Save features to disk
        with open(os.path.join(default_feature_folder, 'arcface-resnet100-512D', 'video' + str(video_id) + '.pkl'), 'wb') as f:
            pickle.dump(video_features_dict, f)

    print('TOTAL ELAPSED TIME: %d minutes %d seconds' %
          ((time.time() - start_t)//60, (time.time() - start_t) % 60))
