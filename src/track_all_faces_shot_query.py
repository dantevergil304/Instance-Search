from track_face_shot_query import getCorrectFrameInShot
from face_extraction import extract_faces_from_image
from ServiceMTCNN import detect_face as lib

import tensorflow as tf
import cv2
import numpy as np
import json
import os
import sys
import glob


def getAllFaceTracks(topic_frame_path, shot_path):
    # Get all frames of shot and topic frame offset
    all_frames, topic_offset = getCorrectFrameInShot(
        topic_frame_path, shot_path)

    # Create MTCNN session
    sess = tf.Session()
    pnet, rnet, onet = lib.create_mtcnn(sess, None)

    # Detect all faces in frames
    all_faces = []
    all_bbs = []
    for frame in all_frames:
        faces, bbs, _ = extract_faces_from_image(frame, pnet, rnet, onet)
        all_faces.append(faces)
        all_bbs.append(bbs)

    print('[+] INITIALIZE PARAMETERS FOR KLT TRACKER')
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    print('[+] DETECT FEATURE POINTS OF FACES IN FIRST FRAME')
    old_topic_frame_gray = cv2.cvtColor(all_frames[0], cv2.COLOR_BGR2GRAY)
    old_topic_frame_gray = cv2.equalizeHist(old_topic_frame_gray)

    # Contain all feature points
    PTS_LIST = []
    PTS_LIST_IDX = []

    # Contain lists of face tracks, elements of each list
    # will be a tuple (frame_offset, face_offset,
    # set of points index that face referring to)
    FACE_TRACKS = []

    for face_offset, bb in enumerate(all_bbs[0]):
        faces_mask = np.zeros_like(old_topic_frame_gray)
        cv2.rectangle(faces_mask, (bb[0], bb[1]),
                      (bb[2], bb[3]), 255, -1)

        p0 = cv2.goodFeaturesToTrack(
            old_topic_frame_gray, mask=faces_mask, **feature_params)

        # For checking if point existed
        pset = dict(zip([tuple(p[0]) for p in PTS_LIST], range(len(PTS_LIST))))

        face_pset = set()
        for p in p0:
            if tuple(p[0]) in pset:
                face_pset.add(pset[tuple(p[0])])
            else:
                PTS_LIST.append(p)
                face_pset.add(len(PTS_LIST)-1)

        FACE_TRACKS.append([(0, face_offset, face_pset)])

    NUM_UNIQUE_PTS = len(PTS_LIST)
    PTS_LIST_IDX = np.arange(NUM_UNIQUE_PTS).tolist()

    print('PTS LIST', PTS_LIST)
    print('NUM UNIQUE PTS', NUM_UNIQUE_PTS)

    # Track faces in next frames
    for frame_offset, (frame, bbs) in enumerate(zip(all_frames[1:], all_bbs[1:])):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        # calculate opitcal flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_topic_frame_gray, frame_gray, np.array(PTS_LIST, np.float32), None, **lk_params)

        PTS_LIST = np.round(p1[st == 1].reshape(-1, 1, 2)).tolist()
        PTS_LIST_IDX = np.array(PTS_LIST_IDX)[st.squeeze() == 1].tolist()

        # detect new feature points
        faces_mask = np.zeros_like(frame_gray)
        for bb in bbs:
            cv2.rectangle(faces_mask, (bb[0], bb[1]), (bb[2], bb[3]), 255, -1)
        new_good_features = cv2.goodFeaturesToTrack(
            frame_gray, mask=faces_mask, **feature_params)

        # For checking if point existed
        pset = dict(zip([tuple(p[0]) for p in PTS_LIST], PTS_LIST_IDX))

        # Add new feature points to points list
        if new_good_features is not None:
            for p in new_good_features:
                if tuple(p[0]) not in pset:
                    PTS_LIST.append(p)
                    PTS_LIST_IDX.append(NUM_UNIQUE_PTS)
                    NUM_UNIQUE_PTS += 1

        # Add to face track
        for face_offset, bb in enumerate(bbs):
            # Check which points is inside bbox
            face_pset = set()
            for i, p in enumerate(PTS_LIST):
                p_idx = PTS_LIST_IDX[i]
                if p[0][0] >= bb[0] and p[0][1] >= bb[1] \
                        and p[0][0] <= bb[2] and p[0][1] <= bb[3]:
                    face_pset.add(p_idx)

            # Check if threshold is large enough for adding to existed face track,
            # or it should belong to a new face track
            thresh = []
            for track in FACE_TRACKS:
                latest_face = track[-1]
                num_shared_points = len(latest_face[2].intersection(face_pset))
                num_all_points = len(latest_face[2].union(face_pset))

                if num_all_points == 0:
                    thresh.append(0)
                else:
                    thresh.append(num_shared_points / num_all_points)

            print(thresh)
            largest_thresh_idx = np.argmax(thresh)
            if thresh[largest_thresh_idx] > 0.3:
                FACE_TRACKS[largest_thresh_idx].append(
                    (frame_offset+1, face_offset, face_pset))
            else:
                FACE_TRACKS.append([(frame_offset+1, face_offset, face_pset)])
        old_topic_frame_gray = frame_gray.copy()

    # Visualize Face Track
    default_face_height = 50
    for idx, track in enumerate(FACE_TRACKS):
        visualize_faces = None
        if len(track) <= 4:
            continue
        for frame_offset, face_offset, _ in track:
            face = all_faces[frame_offset][face_offset]
            height, width = face.shape[:2]
            face_width = int(default_face_height * width / height)

            face = cv2.resize(face, (face_width, default_face_height))

            if visualize_faces is None:
                visualize_faces = face
            else:
                visualize_faces = np.hstack((visualize_faces, face))

        cv2.imshow(f'face track {idx}', visualize_faces)

    cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    with open('../cfg/config.json', 'r') as f:
        cfg = json.load(f)

    query_folder = cfg['raw_data']['queries_folder']
    query_shot_folder = cfg['raw_data']['shot_example_folder']
    info_folder = cfg['raw_data']['info_folder']

    topic_frame_path = os.path.join(query_folder, 'minty.4.src.png')
    topic_mask_path = os.path.join(query_folder, 'minty.4.mask.png')
    shot_path = os.path.join(
        query_shot_folder, 'minty', 'shot0_1155.mp4')

    getAllFaceTracks(topic_frame_path, shot_path)


if __name__ == '__main__':
    main()
