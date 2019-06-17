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
import time
import xml.etree.ElementTree as ET


def splitShotBoundary(all_frames):
    shot_boundaries = [0]
    for frame_offset in range(1, len(all_frames)):
        prev_frame_gray = cv2.cvtColor(all_frames[frame_offset-1], cv2.COLOR_BGR2GRAY )
        frame_gray = cv2.cvtColor(all_frames[frame_offset], cv2.COLOR_BGR2GRAY )

        diff = np.sum((prev_frame_gray - frame_gray) ** 2) / \
              (frame_gray.shape[0] * frame_gray.shape[1])
        if diff > 70:
            shot_boundaries.append(frame_offset) 
    return shot_boundaries


def getAllFaceTracksOfVideoShot(topic_frame_path, all_frames, all_faces, all_bbs, begin, end):
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

    for face_offset, bb in enumerate(all_bbs[begin]):
        faces_mask = np.zeros_like(old_topic_frame_gray)
        cv2.rectangle(faces_mask, (bb[0], bb[1]),
                      (bb[2], bb[3]), 255, -1)

        p0 = cv2.goodFeaturesToTrack(
            old_topic_frame_gray, mask=faces_mask, **feature_params)

        # For checking if point existed
        pset = dict(zip([tuple(p[0]) for p in PTS_LIST], range(len(PTS_LIST))))

        if p0 is not None:
            face_pset = set()
            for p in p0:
                if tuple(p[0]) in pset:
                    face_pset.add(pset[tuple(p[0])])
                else:
                    PTS_LIST.append(p)
                    face_pset.add(len(PTS_LIST)-1)

        FACE_TRACKS.append([(begin, face_offset, face_pset)])

    NUM_UNIQUE_PTS = len(PTS_LIST)
    PTS_LIST_IDX = np.arange(NUM_UNIQUE_PTS).tolist()

    print('PTS LIST', PTS_LIST)
    print("PTS LIST INDEX", PTS_LIST_IDX)
    print('NUM UNIQUE PTS', NUM_UNIQUE_PTS)

    # Track faces in next frames
    for frame_offset, (frame, bbs) in enumerate(zip(all_frames[(begin+1):(end+1)], all_bbs[(begin+1):(end+1)])):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        # calculate opitcal flow
        if PTS_LIST != []:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                old_topic_frame_gray, frame_gray, np.array(PTS_LIST, np.float32), None, **lk_params)

            PTS_LIST = np.round(p1[st == 1].reshape(-1, 1, 2)).tolist()
            PTS_LIST_IDX = np.array(PTS_LIST_IDX)[st.squeeze() == 1].reshape(-1).tolist()

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
        if PTS_LIST != []:
            for face_offset, bb in enumerate(bbs):
                # Check which points is inside bbox
                face_pset = set()
                print("PTS LIST INDEX", PTS_LIST_IDX)
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
                if thresh != []:
                    largest_thresh_idx = np.argmax(thresh)
                else:
                    largest_thresh_idx = None
                if largest_thresh_idx is not None and thresh[largest_thresh_idx] > 0.3:
                    FACE_TRACKS[largest_thresh_idx].append(
                        (begin+frame_offset+1, face_offset, face_pset))
                else:
                    FACE_TRACKS.append([(begin+frame_offset+1, face_offset, face_pset)])
        old_topic_frame_gray = frame_gray.copy()

    # Visualize Face Track
    VISUALIZE_FACE_TRACKS = []
    default_face_height = 50
    print("#Face tracks found", len(FACE_TRACKS))
    for idx, track in enumerate(FACE_TRACKS):
        visualize_faces = None
        # if len(track) <= 4:
        #     continue
        for frame_offset, face_offset, _ in track:
            face = all_faces[frame_offset][face_offset]
            height, width = face.shape[:2]
            face_width = int(default_face_height * width / height)

            face = cv2.resize(face, (face_width, default_face_height))

            if visualize_faces is None:
                visualize_faces = face
            else:
                visualize_faces = np.hstack((visualize_faces, face))
        VISUALIZE_FACE_TRACKS.append(visualize_faces)

        # cv2.imshow(f'face track {idx}', visualize_faces)

    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return FACE_TRACKS, VISUALIZE_FACE_TRACKS


def getAllFaceTrackShotQuery(topic_frame_path, shot_path):
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

    boundaries = splitShotBoundary(all_frames)

    all_face_tracks = []
    all_visualize_face_tracks = []

    num_boundaries = len(boundaries)
    print('Number of boundary shots', num_boundaries)

    # Visualize Boundary shots
    # for idx in range(num_boundaries):
    #     begin = boundaries[idx]
    #     if idx == num_boundaries - 1:
    #         end = len(all_frames) - 1 
    #     else:
    #         end = boundaries[idx+1]-1 
    #     for frame_idx in range(begin, end+1):
    #         resize_frame = cv2.resize(all_frames[frame_idx], None, fx=0.5, fy=0.5)
    #         cv2.imshow(f'boundary shot {idx}', resize_frame) 
    #         cv2.waitKey(1)
    #         
    #     cv2.waitKey(0) 
    #     cv2.destroyAllWindows()
        

    for idx in range(num_boundaries):
        print(f'[+] Processing Boundary {idx}')
        begin = boundaries[idx]
        if idx == num_boundaries - 1:
            end = len(all_frames) - 1 
        else:
            end = boundaries[idx+1]-1 

        face_tracks, visualize_face_tracks = getAllFaceTracksOfVideoShot(topic_frame_path, all_frames, all_faces, all_bbs, begin, end)
        
        all_face_tracks.extend(face_tracks)
        all_visualize_face_tracks.extend(visualize_face_tracks)

    # for idx, face_track in enumerate(all_visualize_face_tracks):
    #     cv2.imshow(f'face track {idx}', face_track)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return all_face_tracks, all_visualize_face_tracks, all_frames, all_faces, all_bbs, topic_offset


def getTopicFaceTrackShotQuery(topic_frame_path, topic_mask_path, shot_path):
    all_tracks, all_visualize_tracks, all_frames, all_faces, all_bbs, topic_offset = getAllFaceTrackShotQuery(topic_frame_path, shot_path) 

    mask_frame = cv2.imread(topic_mask_path, 0)
    mask_frame = cv2.resize(mask_frame, (768, 576))

    print('Topic frame offset', topic_offset)

    tracks_frame_with_largest_overlapped_mask = [np.inf] * len(all_tracks)
 
    for track_idx, track in enumerate(all_tracks):
        best_overlap = 0
        best_overlap_idx = np.inf
        min_topic_offset_distance = np.inf
        for fr_offset, bb_offset, _ in track:
            bb = all_bbs[fr_offset][bb_offset]
            face_mask = np.zeros_like(mask_frame)
            print('bbox', bb)
            cv2.rectangle(face_mask, (bb[0], bb[1]), (bb[2], bb[3]), 255, -1)

            face_mask_num_px = np.count_nonzero(face_mask)
            overlapped_mask_num_px = np.count_nonzero(cv2.bitwise_and(face_mask, mask_frame))
            # if overlapped_mask_num_px / face_mask_num_px >= 0.5 and overlapped_mask_num_px > best_overlap:
            # if overlapped_mask_num_px > best_overlap:
            if overlapped_mask_num_px / face_mask_num_px >= 0.5 and abs(fr_offset-topic_offset) < min_topic_offset_distance:
                best_overlap = overlapped_mask_num_px
                best_overlap_idx = fr_offset
                min_topic_offset_distance = abs(fr_offset-topic_offset)

        tracks_frame_with_largest_overlapped_mask[track_idx] = (min_topic_offset_distance, -best_overlap)
 
    print('tracks frame have with largest overlapped mask', tracks_frame_with_largest_overlapped_mask)
    correct_track_key = tracks_frame_with_largest_overlapped_mask.index(min(tracks_frame_with_largest_overlapped_mask))
    print('correct track key', correct_track_key)


    # check if no track overlapped the mask
    if tracks_frame_with_largest_overlapped_mask[correct_track_key][1] == 0:
        return [], None, topic_offset 

    default_face_height = 50
    visualize_faces = None
    track_bbs = all_tracks[correct_track_key]
    track_faces = []
    for fr_offset, face_offset, _ in track_bbs:
        face = all_faces[fr_offset][face_offset]
        track_faces.append(face)
        height, width = face.shape[:2]
        face_width = int(default_face_height * width / height)

        face = cv2.resize(face, (face_width, default_face_height))

        if visualize_faces is None:
            visualize_faces = face
        else:
            visualize_faces = np.hstack((visualize_faces, face))

    # cv2.imshow('visualize face track', visualize_faces)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return track_faces, visualize_faces, topic_offset



def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    with open('../cfg/config.json', 'r') as f:
        cfg = json.load(f)

    query_folder = cfg['raw_data']['queries_folder']
    query_shot_folder = cfg['raw_data']['shot_example_folder']
    info_folder = cfg['raw_data']['info_folder']

    # topic_frame_path = os.path.join(query_folder, 'denise.4.src.png')
    # topic_mask_path = os.path.join(query_folder, 'denise.4.mask.png')
    # shot_path = os.path.join(
    #     query_shot_folder, 'denise', 'shot236_36.mp4')

    # getTopicFaceTrackShotQuery(topic_frame_path, topic_mask_path, shot_path)

    topic_file = os.path.join(info_folder, 'ins.auto.topics.2019.xml')
    print(topic_file)
    tree = ET.parse(topic_file)
    root = tree.getroot()
 
    info_dict = dict()
    for topic in root.findall('videoInstanceTopic'):
        for image in topic.findall('imageExample'):
            info_dict[image.attrib['src']] = image.attrib['shotID']
 
    names = ['bradley', 'max', 'ian', 'pat', 'denise', 'phil', 'jane', 'jack', 'dot', 'stacey']
    names = ['phil', 'jane', 'jack', 'dot', 'stacey']
    # names = ['pat', 'denise', 'phil', 'jane', 'jack', 'dot', 'stacey']
    for name in names:
        for i in range(1, 5):
            topic_frame_path = os.path.join(
                query_folder, f'{name}.{i}.src.png')
            topic_mask_path = os.path.join(
                query_folder, f'{name}.{i}.mask.png')
            shot_path = os.path.join(
                query_shot_folder, f'{name}', info_dict[f'{name}.{i}.src.png'] + '.mp4')
 
            print('[+] Topic frame path', topic_frame_path)
            print('[+] Topic mask path', topic_mask_path)
            print('[+] Topic shot path', shot_path)
 
            track_faces, visualize_faces, topic_face_index = getTopicFaceTrackShotQuery(topic_frame_path, topic_mask_path, shot_path)
 
            shot_face_folder = os.path.join(
                query_folder, 'shot_query_faces', f'{name}', f'{i}')
            if not os.path.exists(shot_face_folder):
                os.makedirs(shot_face_folder, exist_ok=True)
            for idx, face in enumerate(track_faces):
                cv2.imwrite(os.path.join(shot_face_folder,
                                         f'{name}.{idx}.face.png'), face)
                with open(os.path.join(shot_face_folder, f'topic_face_index.txt'), 'w') as f:
                    f.write(str(topic_face_index))
 
            visualize_face_folder = os.path.join(
                query_folder, 'visualize_shot_query_faces', f'{name}')
            if not os.path.exists(visualize_face_folder):
                os.makedirs(visualize_face_folder, exist_ok=True)
            if visualize_faces is not None:
                cv2.imwrite(os.path.join(visualize_face_folder,
                                         f'facetrack.{i}.png'), visualize_faces)


if __name__ == '__main__':
    main()
