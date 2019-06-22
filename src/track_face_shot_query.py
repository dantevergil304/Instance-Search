from face_extraction_queries import detect_face_by_image
from face_extraction import extract_faces_from_image
from deep_learning_utils import extendBB
from collections import OrderedDict

import cv2
import numpy as np
import json
import os
import sys
import tensorflow as tf
import glob
import xml.etree.ElementTree as ET


sys.path.append(os.path.abspath(os.path.join('..', '3rd_party')))
from ServiceMTCNN import detect_face as lib


def getCorrectFrameInShot(topic_frame_path, shot_path):
    topic_frame = cv2.imread(topic_frame_path)
    topic_frame = cv2.resize(topic_frame, (768, 576))

    cap = cv2.VideoCapture(shot_path)

    all_frames = []
    min_topic_offset = None
    min_diff = np.inf
    current_offset = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if ret is True:
            all_frames.append(frame)
            diff = np.sum((topic_frame - frame) ** 2)
            if diff < min_diff:
                min_topic_offset = current_offset
                min_diff = diff
        else:
            break

        current_offset += 1

    cap.release()
    return all_frames, min_topic_offset


def getFaceTrackOfFace(face_bb, frame_offset, all_faces, all_bbs, all_frames, isForward=True):
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    old_topic_frame_gray = cv2.cvtColor(all_frames[frame_offset], cv2.COLOR_BGR2GRAY)
    old_topic_frame_gray = cv2.equalizeHist(old_topic_frame_gray)
    face_mask = np.zeros_like(old_topic_frame_gray)
    cv2.rectangle(face_mask, (face_bb[0], face_bb[1]),
                  (face_bb[2], face_bb[3]), 255, -1)

    p0 = cv2.goodFeaturesToTrack(
        old_topic_frame_gray, mask=face_mask, **feature_params)

    old_topic_frame_gray_temp = old_topic_frame_gray.copy()
    p0_temp = p0

    tracked_faces = []
    tracked_bbs = []
    # if face is not None:
    #     tracked_faces.append(face)
    #     topic_face_index = 0

    if isForward:
        range_generator = range(frame_offset+1, len(all_frames))
        prev_idx = -1
    else:
        range_generator = range(frame_offset-1, -1, -1)
        prev_idx = 1


    for idx, i in enumerate(range_generator):
        visualize_frame = all_frames[i].copy()
        prev_frame_gray = cv2.cvtColor(all_frames[i+prev_idx], cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(all_frames[i], cv2.COLOR_BGR2GRAY)

        diff = np.sum((prev_frame_gray - frame_gray) ** 2) / \
            (frame_gray.shape[0] * frame_gray.shape[1])
        if diff > 80:
            break

        frame_gray = cv2.equalizeHist(frame_gray)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_topic_frame_gray, frame_gray, p0, None, **lk_params)
        if st is None:
            break

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        features_mask = np.zeros_like(frame_gray, dtype=np.uint8)
        for new, old in zip(good_new, good_old):
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()
            cv2.circle(visualize_frame, (x_new, y_new), 2, (0, 255, 0), -1)
            cv2.circle(features_mask, (x_new, y_new), 0, 255, 0)
        # cv2.imshow('feature mask', features_mask)

        faces_mask = np.zeros_like(frame_gray, dtype=np.uint8)
        for bb in all_bbs[i]:
            left, top, right, bottom = extendBB(
                frame_gray.shape[:2], bb[0], bb[1], bb[2], bb[3], 0)
            cv2.rectangle(faces_mask, (left, top), (right, bottom), 255, -1)
        # cv2.imshow('faces mask', faces_mask)

        features_in_bb_mask = cv2.bitwise_and(features_mask, faces_mask)
        # cv2.imshow('feature in bb', features_in_bb_mask)

        features_in_bb_coord = np.where(features_in_bb_mask == 255)
        selectedFace = None
        selectedBB = None
        for px in zip(features_in_bb_coord[1], features_in_bb_coord[0]):
            x, y = px
            for face, bb in zip(all_faces[i], all_bbs[i]):
                if x >= bb[0] and x <= bb[2] and y >= bb[1] and y <= bb[3]:
                    selectedFace = face
                    selectedBB = bb

        if selectedBB is not None:
            cv2.rectangle(visualize_frame, (selectedBB[0], selectedBB[1]), (
                selectedBB[2], selectedBB[3]), (0, 255, 0), 2)
            tracked_faces.insert(0, selectedFace)
            tracked_bbs.insert(0, (i, selectedBB))
            # topic_face_index += 1
        # cv2.imshow('forward tracking', visualize_frame)

        # mask_img = np.hstack((features_mask, faces_mask))
        # result_img = np.hstack((features_in_bb_mask, frame_gray))
        # visualize_img = np.vstack((mask_img, result_img))
        # cv2.imshow('visualize img', visualize_img)

        # cv2.waitKey()
        # cv2.destroyAllWindows()

        old_topic_frame_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)


    return tracked_faces, tracked_bbs

def getFaceTrackOfFaceBackAndForth(face, face_bb, frame_offset, all_faces, all_bbs, all_frames):
    forward_tracked_faces, forward_tracked_bbs = getFaceTrackOfFace(face_bb, frame_offset, all_faces, all_bbs, all_frames)
    backward_tracked_faces, backward_tracked_bbs = getFaceTrackOfFace(face_bb, frame_offset, all_faces, all_bbs, all_frames, isForward=False)

    tracked_faces = backward_tracked_faces + [face] + forward_tracked_faces
    tracked_bbs = backward_tracked_bbs + [(frame_offset, face_bb)] + forward_tracked_bbs


    if len(tracked_faces) > 0:
        default_face_height = 50
        visualize_faces = None
        for face in tracked_faces:
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

    return tracked_faces, tracked_bbs


def getFaceTracksInShot(topic_frame_path, topic_mask_path, shot_path):
    all_frames, topic_offset = getCorrectFrameInShot(
        topic_frame_path, shot_path)

    # Create session for MTCNN
    sess = tf.Session()
    pnet, rnet, onet = lib.create_mtcnn(sess, None)

    all_faces = []
    all_bbs = []
    all_scores = []
    for frame in all_frames:
        faces, bbs, scores = extract_faces_from_image(frame, pnet, rnet, onet)
        all_faces.append(faces)
        all_bbs.append(bbs)
        all_scores.append(scores)

    all_tracked_faces = []
    all_tracked_bbs = []
    print("Numbers of frames", len(all_frames))
    for frame_offset, (faces, bbs, frame, scores) in enumerate(zip(all_faces, all_bbs, all_frames, all_scores)):
        print(frame_offset, end=' ')
        for face, bb, score in zip(faces, bbs, scores):
            tracked_faces, tracked_bbs = getFaceTrackOfFaceBackAndForth(face, bb, frame_offset, all_faces, all_bbs, all_frames)
            all_tracked_faces.append(tracked_faces)
            all_tracked_bbs.append(tracked_bbs)
    

    return all_tracked_faces, all_tracked_bbs, all_frames, topic_offset


def removeOverlappedTracks(all_tracked_faces, all_tracked_bbs):
    removed_tracks = [0] * len(all_tracked_faces)
    all_tracks = OrderedDict()
    track_offset = 0

    for cur_idx in range(len(all_tracked_faces)):
        if removed_tracks[cur_idx] == 1:
            continue

        # print(all_tracked_bbs[cur_idx])
        current_bbs_set = set(all_tracked_bbs[cur_idx])
        current_bbs_dict = dict(zip(all_tracked_bbs[cur_idx], all_tracked_faces[cur_idx]))

        face_track = None
        for next_idx in range(cur_idx+1, len(all_tracked_faces)):
            if removed_tracks[next_idx] == 1:
                continue

            next_bbs_set = set(all_tracked_bbs[next_idx])
            next_bbs_dict = dict(zip(all_tracked_bbs[next_idx], all_tracked_faces[next_idx]))

            intersection = current_bbs_set.intersection(next_bbs_set)
            if len(intersection) / len(current_bbs_set) >= 0.2 and len(intersection) / len(next_bbs_set) >= 0.2:
                current_bbs_set = current_bbs_set.union(next_bbs_set)  
                removed_tracks[next_idx] = 1

                if face_track is None:
                    face_track = {**current_bbs_dict, **next_bbs_dict}
                else:
                    face_track = {**face_track, **current_bbs_dict, **next_bbs_dict}

        if face_track is not None:
            all_tracks[track_offset] = face_track
            track_offset += 1

    for track in all_tracks:
        if len(all_tracks[track]) > 0:
            default_face_height = 50
            visualize_faces = None
            track_bbs = sorted(all_tracks[track], key=lambda x: x[0])
            for face in track_bbs:
                face = all_tracks[track][face]
                height, width = face.shape[:2]
                face_width = int(default_face_height * width / height)

                face = cv2.resize(face, (face_width, default_face_height))

                if visualize_faces is None:
                    visualize_faces = face
                else:
                    visualize_faces = np.hstack((visualize_faces, face))

            cv2.imshow('remove overlapped face track', visualize_faces)
            cv2.waitKey()
            cv2.destroyAllWindows()

    return all_tracks
        

def getCorrectFaceTrackInShotType2(topic_frame_path, topic_mask_path, shot_path):
    all_tracked_faces, all_tracked_bbs, all_frames, topic_offset = getFaceTracksInShot(topic_frame_path, topic_mask_path, shot_path)
    mask_frame = cv2.imread(topic_mask_path, 0)
    mask_frame = cv2.resize(mask_frame, (768, 576))

    print('Topic frame offset', topic_offset)

    all_tracks = removeOverlappedTracks(all_tracked_faces, all_tracked_bbs)
    tracks_frame_with_largest_overlapped_mask = [np.inf] * len(all_tracks)

    for track in all_tracks:
        track_bbs = sorted(all_tracks[track], key=lambda x: x[0])
        best_overlap = 0
        best_overlap_idx = np.inf 
        min_topic_offset_distance = np.inf
        for fr, bb in track_bbs:
            face_mask = np.zeros_like(mask_frame)
            cv2.rectangle(face_mask, (bb[0], bb[1]), (bb[2], bb[3]), 255, -1)

            face_mask_num_px = np.count_nonzero(face_mask)
            overlapped_mask_num_px = np.count_nonzero(cv2.bitwise_and(face_mask, mask_frame))
            # if overlapped_mask_num_px / face_mask_num_px >= 0.5 and overlapped_mask_num_px > best_overlap:
            # if overlapped_mask_num_px > best_overlap:
            if overlapped_mask_num_px / face_mask_num_px >= 0.5 and abs(fr-topic_offset) < min_topic_offset_distance:
                best_overlap = overlapped_mask_num_px
                best_overlap_idx = fr
                min_topic_offset_distance = abs(fr-topic_offset)

        tracks_frame_with_largest_overlapped_mask[track] = (min_topic_offset_distance, -best_overlap)
    
    print(tracks_frame_with_largest_overlapped_mask)
    # distance_from_topic_offset = np.absolute(tracks_frame_with_largest_overlapped_mask - topic_offset)
    # tracks_frame_with_largest_overlapped_mask = min(tracks_frame_with_largest_overlapped_mask)
    # print(distance_from_topic_offset)
    # correct_track_key = np.argmin(distance_from_topic_offset)
    correct_track_key = tracks_frame_with_largest_overlapped_mask.index(min(tracks_frame_with_largest_overlapped_mask))
    print('correct track key', correct_track_key)

    default_face_height = 50
    visualize_faces = None
    track_bbs = sorted(all_tracks[correct_track_key], key=lambda x: x[0])
    track_faces = []
    for face in track_bbs:
        face = all_tracks[correct_track_key][face]
        track_faces.append(face)
        height, width = face.shape[:2]
        face_width = int(default_face_height * width / height)

        face = cv2.resize(face, (face_width, default_face_height))

        if visualize_faces is None:
            visualize_faces = face
        else:
            visualize_faces = np.hstack((visualize_faces, face))

    cv2.imshow('visualize face track', visualize_faces)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return track_faces, visualize_faces, topic_offset


def getCorrectFaceTrackInShot(topic_frame_path, topic_mask_path, shot_path):
    all_frames, topic_offset = getCorrectFrameInShot(
        topic_frame_path, shot_path)

    # Create session for MTCNN
    sess = tf.Session()
    pnet, rnet, onet = lib.create_mtcnn(sess, None)

    all_faces = []
    all_bbs = []
    for frame in all_frames:
        faces, bbs = extract_faces_from_image(frame, pnet, rnet, onet)
        all_faces.append(faces)
        all_bbs.append(bbs)

    topic_frame = cv2.imread(topic_frame_path)
    mask_frame = cv2.imread(topic_mask_path, 0)
    # mask_frame = cv2.resize(mask_frame, (768, 576))

    _, bb, _ = detect_face_by_image([topic_frame], [mask_frame])
    # face = face[0]
    bb = list(bb[0])
    bb[0] /= (topic_frame.shape[1] / 768)
    bb[1] /= (topic_frame.shape[0] / 576)
    bb[2] /= (topic_frame.shape[1] / 768)
    bb[3] /= (topic_frame.shape[0] / 576)
    bb = [int(c) for c in bb]
    topic_frame = cv2.resize(topic_frame, (768, 576))
    mask_frame = cv2.resize(mask_frame, (768, 576))
    face = topic_frame[bb[1]:bb[3], bb[0]:bb[2]]

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    old_topic_frame_gray = cv2.cvtColor(topic_frame, cv2.COLOR_BGR2GRAY)
    old_topic_frame_gray = cv2.equalizeHist(old_topic_frame_gray)
    face_mask = np.zeros_like(old_topic_frame_gray)
    cv2.rectangle(face_mask, (bb[0], bb[1]), (bb[2], bb[3]), 255, -1)

    # face_mask = cv2.bitwise_and(face_mask, mask_frame)

    p0 = cv2.goodFeaturesToTrack(
        old_topic_frame_gray, mask=face_mask, **feature_params)

    old_topic_frame_gray_temp = old_topic_frame_gray.copy()
    p0_temp = p0

    tracked_faces = []
    if face is not None:
        tracked_faces.append(face)
        topic_face_index = 0

    # Backward Tracking
    for idx, i in enumerate(range(topic_offset-1, -1, -1)):
        visualize_frame = all_frames[i].copy()
        prev_frame_gray = cv2.cvtColor(all_frames[i+1], cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(all_frames[i], cv2.COLOR_BGR2GRAY)

        diff = np.sum((prev_frame_gray - frame_gray) ** 2) / \
            (frame_gray.shape[0] * frame_gray.shape[1])
        if diff > 80:
            break

        frame_gray = cv2.equalizeHist(frame_gray)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_topic_frame_gray, frame_gray, p0, None, **lk_params)
        if st is None:
            break

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        features_mask = np.zeros_like(frame_gray, dtype=np.uint8)
        for new, old in zip(good_new, good_old):
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()
            cv2.circle(visualize_frame, (x_new, y_new), 2, (0, 255, 0), -1)
            cv2.circle(features_mask, (x_new, y_new), 0, 255, 0)
        # cv2.imshow('feature mask', features_mask)

        faces_mask = np.zeros_like(frame_gray, dtype=np.uint8)
        for bb in all_bbs[i]:
            left, top, right, bottom = extendBB(
                frame_gray.shape[:2], bb[0], bb[1], bb[2], bb[3], 0)
            cv2.rectangle(faces_mask, (left, top), (right, bottom), 255, -1)
        # cv2.imshow('faces mask', faces_mask)

        features_in_bb_mask = cv2.bitwise_and(features_mask, faces_mask)
        # cv2.imshow('feature in bb', features_in_bb_mask)

        features_in_bb_coord = np.where(features_in_bb_mask == 255)
        selectedFace = None
        selectedBB = None
        for px in zip(features_in_bb_coord[1], features_in_bb_coord[0]):
            x, y = px
            for face, bb in zip(all_faces[i], all_bbs[i]):
                if x >= bb[0] and x <= bb[2] and y >= bb[1]and y <= bb[3]:
                    selectedFace = face
                    selectedBB = bb

        if selectedBB is not None:
            cv2.rectangle(visualize_frame, (selectedBB[0], selectedBB[1]), (
                selectedBB[2], selectedBB[3]), (0, 255, 0), 2)
            tracked_faces.insert(0, selectedFace)
            topic_face_index += 1
        # cv2.imshow('forward tracking', visualize_frame)

        # mask_img = np.hstack((features_mask, faces_mask))
        # result_img = np.hstack((features_in_bb_mask, frame_gray))
        # visualize_img = np.vstack((mask_img, result_img))
        # cv2.imshow('visualize img', visualize_img)

        # cv2.waitKey()
        # cv2.destroyAllWindows()

        old_topic_frame_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    # Forward Tracking
    old_topic_frame_gray = old_topic_frame_gray_temp
    p0 = p0_temp
    for idx, i in enumerate(range(topic_offset+1, len(all_frames))):
        visualize_frame = all_frames[i].copy()
        prev_frame_gray = cv2.cvtColor(all_frames[i-1], cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(all_frames[i], cv2.COLOR_BGR2GRAY)

        diff = np.sum((prev_frame_gray - frame_gray) ** 2) / \
            (frame_gray.shape[0] * frame_gray.shape[1])
        if diff > 80:
            break

        frame_gray = cv2.equalizeHist(frame_gray)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_topic_frame_gray, frame_gray, p0, None, **lk_params)
        if st is None:
            break

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        features_mask = np.zeros_like(frame_gray, dtype=np.uint8)
        for new, old in zip(good_new, good_old):
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()
            # cv2.circle(frame_gray, (x_new, y_new), 2, (0, 255, 0), -1)
            cv2.circle(features_mask, (x_new, y_new), 0, 255, 0)
        # cv2.imshow('feature mask', features_mask)

        faces_mask = np.zeros_like(frame_gray, dtype=np.uint8)
        for bb in all_bbs[i]:
            left, top, right, bottom = extendBB(
                frame_gray.shape[:2], bb[0], bb[1], bb[2], bb[3], 0)
            cv2.rectangle(faces_mask, (left, top), (right, bottom), 255, -1)
        # cv2.imshow('faces mask', faces_mask)

        features_in_bb_mask = cv2.bitwise_and(features_mask, faces_mask)
        # cv2.imshow('feature in bb', features_in_bb_mask)

        features_in_bb_coord = np.where(features_in_bb_mask == 255)
        selectedFace = None
        selectedBB = None
        for px in zip(features_in_bb_coord[1], features_in_bb_coord[0]):
            x, y = px
            for face, bb in zip(all_faces[i], all_bbs[i]):
                if x >= bb[0] and x <= bb[2] and y >= bb[1]and y <= bb[3]:
                    selectedFace = face
                    selectedBB = bb

        if selectedBB is not None:
            cv2.rectangle(visualize_frame, (selectedBB[0], selectedBB[1]), (
                selectedBB[2], selectedBB[3]), (0, 255, 0), 2)
            tracked_faces.append(selectedFace)
        # cv2.imshow('forward tracking', visualize_frame)

        # mask_img = np.hstack((features_mask, faces_mask))
        # result_img = np.hstack((features_in_bb_mask, frame_gray))
        # visualize_img = np.vstack((mask_img, result_img))
        # cv2.imshow('visualize img', visualize_img)

        # cv2.waitKey()
        # cv2.destroyAllWindows()

        old_topic_frame_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    default_face_height = 50
    visualize_faces = None
    for face in tracked_faces:
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
    return tracked_faces, visualize_faces, topic_face_index


def main_track_type1():
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    with open('../cfg/config.json', 'r') as f:
        cfg = json.load(f)

    query_folder = cfg['raw_data']['queries_folder']
    query_shot_folder = cfg['raw_data']['shot_example_folder']
    info_folder = cfg['raw_data']['info_folder']

    # test_topic_frame_path = os.path.join(query_folder, 'heather.1.src.png')
    # test_topic_mask_path = os.path.join(query_folder, 'heather.1.mask.png')
    # test_shot_path = os.path.join(
    #     query_shot_folder, 'heather', 'shot0_769.mp4')
    # topic_frame = cv2.imread(test_topic_frame_path)
    # topic_frame = cv2.resize(topic_frame, (768, 576))

    # all_frames, topic_offset = getCorrectFrameInShot(
    #     test_topic_frame_path, test_shot_path)

    # # cv2.imshow('found frame', all_frames[topic_offset])
    # # cv2.imshow('gt frame', topic_frame)
    # # cv2.waitKey()
    # # cv2.destroyAllWindows()
    # getCorrectFaceTrackInShot(test_topic_frame_path,
    #                           test_topic_mask_path, test_shot_path)

    topic_file = os.path.join(info_folder, 'ins.auto.topics.2018.xml')
    print(topic_file)
    tree = ET.parse(topic_file)
    root = tree.getroot()

    info_dict = dict()
    for topic in root.findall('videoInstanceTopic'):
        for image in topic.findall('imageExample'):
            info_dict[image.attrib['src']] = image.attrib['shotID']

    names = ['chelsea', 'darrin', 'garry', 'heather',
             'jane', 'jack', 'max', 'minty', 'mo', 'zainab']
    names = ['jack', 'max', 'minty', 'mo', 'zainab']
    # names = ['chelsea']
    for name in names:
        for i in range(1, 5):
            topic_frame_path = os.path.join(
                query_folder, f'{name}.{i}.src.png')
            topic_mask_path = os.path.join(
                query_folder, f'{name}.{i}.mask.png')
            shot_path = os.path.join(
                query_shot_folder, f'{name}', info_dict[f'{name}.{i}.src.png'] + '.mp4')

            print(topic_frame_path, topic_mask_path, shot_path)

            track_faces, visualize_faces, topic_face_index = getCorrectFaceTrackInShot(
                topic_frame_path, topic_mask_path, shot_path)

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
                os.mkdir(visualize_face_folder)
            cv2.imwrite(os.path.join(visualize_face_folder,
                                     f'facetrack.{i}.png'), visualize_faces)


def main_track_type2():
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    with open('../cfg/config.json', 'r') as f:
        cfg = json.load(f)

    query_folder = cfg['raw_data']['queries_folder']
    query_shot_folder = cfg['raw_data']['shot_example_folder']
    info_folder = cfg['raw_data']['info_folder']

    topic_frame_path = os.path.join(query_folder, 'chelsea.3.src.png')
    topic_mask_path = os.path.join(query_folder, 'chelsea.3.mask.png')
    shot_path = os.path.join(
        query_shot_folder, 'chelsea', 'shot0_1146.mp4')


    # all_tracked_faces, all_tracked_bbs = getFaceTracksInShot(topic_frame_path, topic_mask_path, shot_path)
    # removeOverlappedTracks(all_tracked_faces, all_tracked_bbs)
    getCorrectFaceTrackInShotType2(topic_frame_path, topic_mask_path, shot_path)

    # topic_file = os.path.join(info_folder, 'ins.auto.topics.2019.xml')
    # print(topic_file)
    # tree = ET.parse(topic_file)
    # root = tree.getroot()

    # info_dict = dict()
    # for topic in root.findall('videoInstanceTopic'):
    #     for image in topic.findall('imageExample'):
    #         info_dict[image.attrib['src']] = image.attrib['shotID']

    # names = ['bradley', 'max', 'ian', 'pat', 'denise', 'phil', 'jane', 'jack', 'dot', 'stacey']
    # names = ['bradley']
    # for name in names:
    #     for i in range(1, 5):
    #         topic_frame_path = os.path.join(
    #             query_folder, f'{name}.{i}.src.png')
    #         topic_mask_path = os.path.join(
    #             query_folder, f'{name}.{i}.mask.png')
    #         shot_path = os.path.join(
    #             query_shot_folder, f'{name}', info_dict[f'{name}.{i}.src.png'] + '.mp4')

    #         print(topic_frame_path)
    #         print(topic_mask_path)
    #         print(shot_path)
    #         track_faces, visualize_faces, topic_face_index = getCorrectFaceTrackInShotType2(topic_frame_path, topic_mask_path, shot_path)

    #         shot_face_folder = os.path.join(
    #             query_folder, 'shot_query_faces', f'{name}', f'{i}')
    #         if not os.path.exists(shot_face_folder):
    #             os.makedirs(shot_face_folder, exist_ok=True)
    #         for idx, face in enumerate(track_faces):
    #             cv2.imwrite(os.path.join(shot_face_folder,
    #                                      f'{name}.{idx}.face.png'), face)
    #             with open(os.path.join(shot_face_folder, f'topic_face_index.txt'), 'w') as f:
    #                 f.write(str(topic_face_index))

    #         visualize_face_folder = os.path.join(
    #             query_folder, 'visualize_shot_query_faces', f'{name}')
    #         if not os.path.exists(visualize_face_folder):
    #             os.mkdir(visualize_face_folder)
    #         cv2.imwrite(os.path.join(visualize_face_folder,
    #                                  f'facetrack.{i}.png'), visualize_faces)


if __name__ == '__main__':
    main_track_type2()
